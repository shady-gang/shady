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
} Context;

static void search_op_for_free_variables(Context* visitor, NodeClass class, String op_name, const Node* node) {
    assert(node);
    switch (node->tag) {
        case Let_TAG: {
            Nodes variables = node->payload.let.variables;
            for (size_t j = 0; j < variables.count; j++) {
                const Node* var = variables.nodes[j];
                bool r = insert_set_get_result(const Node*, visitor->current_scope->bound_set, var);
                assert(r);
            }
            break;
        }
        case Variablez_TAG:
        case Param_TAG: {
            const Node** found = find_key_dict(const Node*, visitor->current_scope->bound_set, node);
            if (!found)
                insert_set_get_result(const Node*, visitor->current_scope->free_set, node);
            return;
        }
        case Function_TAG:
        case Case_TAG:
        case BasicBlock_TAG: assert(false);
        default: break;
    }
    visit_node_operands(&visitor->visitor, IGNORE_ABSTRACTIONS_MASK | NcVariable, node);
}

static CFNodeVariables* create_node_variables(CFNode* cfnode) {
    CFNodeVariables* v = calloc(sizeof(CFNodeVariables), 1);
    *v = (CFNodeVariables) {
        .node = cfnode,
        .free_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    return v;
}

static CFNodeVariables* visit_domtree(Context* ctx, CFNode* cfnode, int depth, CFNodeVariables* parent) {
    Context new_context = *ctx;
    ctx = &new_context;

    ctx->current_scope = create_node_variables(cfnode);
    ctx->current_scope->bound_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    insert_dict(CFNode*, CFNodeVariables*, ctx->map, cfnode, ctx->current_scope);
    const Node* abs = cfnode->node;

    bool is_named = abs->tag != Case_TAG;

    // Bind parameters
    Nodes params = get_abstraction_params(abs);
    for (size_t j = 0; j < params.count; j++) {
        const Node* param = params.nodes[j];
        bool r = insert_set_get_result(const Node*, ctx->current_scope->bound_set, param);
        assert(r);
    }

    const Node* body = get_abstraction_body(abs);
    if (body)
        visit_op(&ctx->visitor, NcTerminator, "body", body);

    for (size_t i = 0; i < entries_count_list(cfnode->dominates); i++) {
        CFNode* child = read_list(CFNode*, cfnode->dominates)[i];
        CFNodeVariables* child_variables = visit_domtree(ctx, child, depth + (is_named ? 1 : 1), ctx->current_scope);
        size_t j = 0;
        const Node* free_var;
        while (dict_iter(child_variables->free_set, &j, &free_var, NULL)) {
            const Node** found = find_key_dict(const Node*, ctx->current_scope->bound_set, free_var);
            if (!found)
                insert_set_get_result(const Node*, ctx->current_scope->free_set, free_var);
            next:;
        }
    }

    if (parent) {
        size_t j = 0;
        const Node* bound;
        while (dict_iter(parent->bound_set, &j, &bound, NULL)) {
            insert_set_get_result(const Node*, ctx->current_scope->bound_set, bound);
        }
    }

    /*String abs_name = get_abstraction_name_unsafe(abs);
    for (int i = 0; i < depth; i++)
        debugvv_print(" ");
    if (abs_name)
        debugvv_print("%s: ", abs_name);
    else
        debugvv_print("%%%d: ", abs->id);

    if (true) {
        bool prev = false;
        size_t i = 0;
        const Node* free_var;
        while (dict_iter(ctx->current_scope->free_set, &i, &free_var, NULL)) {
            if (prev) {
                debugvv_print(", ");
            }
            log_node(DEBUGVV, free_var);
            prev = true;
        }
    }

    if (true) {
        debugvv_print(". Binds: ");
        bool prev = false;
        for (size_t i = 0; i < params.count; i++) {
            if (prev) {
                debugvv_print(", ");
            }
            log_node(DEBUGVV, params.nodes[i]);
            prev = true;
        }
    }

    debugvv_print("\n");*/

    // Unbind parameters
    for (size_t j = 0; j < params.count; j++) {
        const Node* param = params.nodes[j];
        assert(find_key_dict(const Node*, ctx->current_scope->bound_set, param));
        //bool r = remove_dict(const Node*, ctx->current_scope->bound_set, param);
        //assert(r);
    }

    return ctx->current_scope;
}

struct Dict* compute_cfg_variables_map(const CFG* cfg) {
    Context ctx = {
        .visitor = {
            .visit_op_fn = (VisitOpFn) search_op_for_free_variables,
        },
        .map = new_dict(CFNode*, CFNodeVariables*, (HashFn) hash_ptr, (CmpFn) compare_ptrs),
    };

    debugv_print("Computing free variables for function '%s' ...\n", get_abstraction_name(cfg->entry->node));
    visit_domtree(&ctx, cfg->entry, 0, NULL);
    return ctx.map;
}

void destroy_cfg_variables_map(struct Dict* map) {
    size_t i = 0;
    CFNodeVariables* value;
    while (dict_iter(map, &i, NULL, &value)) {
        destroy_dict(value->bound_set);
        destroy_dict(value->free_set);
        free((void*) value);
    }
    destroy_dict(map);
}
