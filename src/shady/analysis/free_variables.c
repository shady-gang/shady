#include "free_variables.h"

#include "log.h"
#include "../visit.h"

#include "../analysis/cfg.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Visitor visitor;
    struct Dict* map;
    CFNodeVariables* current_scope;
    struct Dict* bound;
    struct Dict* live;
    CfgVariablesAnalysisFlags flags;
} Context;

static void copy_set(struct Dict* dst, struct Dict* src) {
    size_t j = 0;
    const Node* item;
    while (dict_iter(src, &j, &item, NULL)) {
        insert_set_get_result(const Node*, dst, item);
    }
}

void dump_set(struct Dict* set) {
    bool prev = false;
    size_t i = 0;
    const Node* bound_var;
    while (dict_iter(set, &i, &bound_var, NULL)) {
        if (prev) {
            debugvv_print(", ");
        }
        log_node(DEBUGVV, bound_var);
        prev = true;
    }
}

static void dump_free_variables(Context* ctx, CFNode* cfnode, int depth) {
    const Node* abs = cfnode->node;
    String abs_name = get_abstraction_name_unsafe(abs);
    for (int i = 0; i < depth; i++)
        debugvv_print(" ");
    if (abs_name)
        debugvv_print("%s: ", abs_name);
    else
        debugvv_print("%%%d: ", abs->id);
    debugvv_print(".");

    if (true) {
        if(ctx->current_scope->free_set) {
            debugvv_print(" Free: [");
            dump_set(ctx->current_scope->free_set);
            debugvv_print("]");
        }
        if(ctx->current_scope->bound_by_dominators_set) {
            debugvv_print(" BoundDom: [");
            dump_set(ctx->current_scope->bound_by_dominators_set);
            debugvv_print("]");
        }
        if(ctx->bound) {
            debugvv_print(" Bound: [");
            dump_set(ctx->bound);
            debugvv_print("]");
        }
        //if(ctx->live) {
        //    debugvv_print(" Live: [");
        //    dump_set(ctx->live);
        //    debugvv_print("]");
        //}
    }

    debugvv_print("\n");
}

static void search_op_for_free_variables(Context* visitor, NodeClass class, String op_name, const Node* node) {
    assert(node);
    switch (node->tag) {
        case Let_TAG: {
            const Node* instr = get_let_instruction(node);
            bool r = insert_set_get_result(const Node*, visitor->bound, instr);
            //assert(r);
            break;
        }
        case Function_TAG:
        case BasicBlock_TAG: assert(false);
        default: break;
    }
    if (node->tag == Param_TAG || is_instruction(node))
        insert_set_get_result(const Node*, visitor->live, node);
    visit_node_operands(&visitor->visitor, IGNORE_ABSTRACTIONS_MASK, node);
}

static CFNodeVariables* create_node_variables(CFNode* cfnode) {
    CFNodeVariables* v = calloc(sizeof(CFNodeVariables), 1);
    *v = (CFNodeVariables) {
        .node = cfnode,
    };
    return v;
}

static Context visit_domtree(Context* parent_context, CFNode* cfnode, int depth) {
    Context context = *parent_context;

    context.current_scope = create_node_variables(cfnode);
    //if (depth > 0)
    //    context.bound = clone_dict(parent_context->bound);
    //else
    context.bound = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    context.live = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);

    insert_dict(CFNode*, CFNodeVariables*, context.map, cfnode, context.current_scope);
    if (context.flags & CfgVariablesAnalysisFlagDomBoundSet) {
        if (depth == 0)
            context.current_scope->bound_by_dominators_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
        else
            context.current_scope->bound_by_dominators_set = clone_dict(parent_context->bound);
    }
    if (context.flags & CfgVariablesAnalysisFlagFreeSet)
        context.current_scope->free_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    const Node* abs = cfnode->node;

    // Bind parameters
    Nodes params = get_abstraction_params(abs);
    for (size_t j = 0; j < params.count; j++) {
        const Node* param = params.nodes[j];
        bool r = insert_set_get_result(const Node*, context.bound, param);
        assert(r);
    }

    const Node* body = get_abstraction_body(abs);
    if (body)
        visit_op(&context.visitor, NcTerminator, "body", body);

    if (context.flags & CfgVariablesAnalysisFlagFreeSet) {
        size_t j = 0;
        const Node* pfv;
        while (dict_iter(context.live, &j, &pfv, NULL)) {
            const Node** found = find_key_dict(const Node*, context.bound, pfv);
            if (found)
                continue;
            insert_set_get_result(const Node*, context.current_scope->free_set, pfv);
        }
    }

    struct Dict* stuff_bound_here = NULL;
    // if we want to compute free variables, we need to keep arround a copy of what's bound in this abstraction
    // and check the free variables against that rather than the transitively bound set we're computing right after
    if (context.flags & CfgVariablesAnalysisFlagFreeSet)
        stuff_bound_here = clone_dict(context.bound);

    if (depth > 0)
        copy_set(context.bound, parent_context->bound);

    for (size_t i = 0; i < entries_count_list(cfnode->dominates); i++) {
        CFNode* child = read_list(CFNode*, cfnode->dominates)[i];
        Context child_context = visit_domtree(&context, child, depth + 1);
        // what's live in children is live to us, too
        copy_set(context.live, child_context.live);
        destroy_dict(child_context.bound);
        destroy_dict(child_context.live);

        if (context.flags & CfgVariablesAnalysisFlagFreeSet) {
            size_t j = 0;
            const Node* pfv;
            while (dict_iter(child_context.current_scope->free_set, &j, &pfv, NULL)) {
                const Node** found = find_key_dict(const Node*, stuff_bound_here, pfv);
                if (found)
                    continue;
                insert_set_get_result(const Node*, context.current_scope->free_set, pfv);
            }
        }
    }

    if (stuff_bound_here)
        destroy_dict(stuff_bound_here);

    if (context.flags & CfgVariablesAnalysisFlagBoundSet)
        context.current_scope->bound_set = clone_dict(context.bound);
    if (context.flags & CfgVariablesAnalysisFlagLiveSet)
        context.current_scope->live_set = clone_dict(context.live);

    //dump_free_variables(&context, cfnode, depth);

    return context;
}

struct Dict* compute_cfg_variables_map(const CFG* cfg, CfgVariablesAnalysisFlags flags) {
    Context root_context = {
        .visitor = {
            .visit_op_fn = (VisitOpFn) search_op_for_free_variables,
        },
        .map = new_dict(CFNode*, CFNodeVariables*, (HashFn) hash_ptr, (CmpFn) compare_ptrs),
        .flags = flags,
    };

    debugv_print("Computing free variables for function '%s' ...\n", get_abstraction_name(cfg->entry->node));
    root_context = visit_domtree(&root_context, cfg->entry, 0);
    destroy_dict(root_context.bound);
    destroy_dict(root_context.live);
    return root_context.map;
}

static void destroy_variables_node(CFNodeVariables* value) {
    if (value->bound_by_dominators_set)
        destroy_dict(value->bound_by_dominators_set);
    if (value->bound_set)
        destroy_dict(value->bound_set);
    if (value->live_set)
        destroy_dict(value->live_set);
    if (value->free_set)
        destroy_dict(value->free_set);
    free((void*) value);
}

void destroy_cfg_variables_map(struct Dict* map) {
    size_t i = 0;
    CFNodeVariables* value;
    while (dict_iter(map, &i, NULL, &value)) {
        destroy_variables_node(value);
    }
    destroy_dict(map);
}
