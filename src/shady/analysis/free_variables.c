#include "free_variables.h"

#include "log.h"
#include "../visit.h"

#include "../analysis/scope.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Visitor visitor;
    struct Dict* ignore_set;
    struct List* free_list;
} Context;

static void visit_fv(Context* visitor, const Node* node) {
    assert(node);
    switch (node->tag) {
        case Variable_TAG: {
            // if we encounter a node we haven't ignored already, it is deemed free
            if (insert_set_get_result(const Node*, visitor->ignore_set, node))
                append_list(const Node*, visitor->free_list, node);
            break;
        }
        case Function_TAG: break; // we do not visit the insides of functions/basic blocks, that's what the domtree search is already doing!
        default: visit_children(&visitor->visitor, node); break;
    }
}

static void visit_domtree(Context* ctx, CFNode* cfnode, int depth) {
    const Function* fun = &cfnode->location.head->payload.fn;

    for (int i = 0; i < depth; i++)
        debug_print(" ");
    debug_print("%s\n", fun->name);

    // Bind parameters
    for (size_t j = 0; j < fun->params.count; j++) {
        const Node* param = fun->params.nodes[j];
        bool r = insert_set_get_result(const Node*, ctx->ignore_set, param);
        assert(r);
    }

    const Body* entry_body = &fun->body->payload.body;
    assert(fun->body);
    for (size_t j = 0; j < entry_body->instructions.count; j++) {
        const Node* instr = entry_body->instructions.nodes[j];
        const Node* actual_instr = instr->tag == Let_TAG ? instr->payload.let.instruction : instr;
        visit_fv(ctx, actual_instr);
        if (instr->tag == Let_TAG) {
            // after being computed, outputs are no longer considered free
            Nodes outputs = instr->payload.let.variables;
            for (size_t k = 0; k < outputs.count; k++) {
                const Node* output = outputs.nodes[k];
                bool r = insert_set_get_result(const Node*, ctx->ignore_set, output);
                assert(r);
            }
        }
    }

    visit_fv(ctx, entry_body->terminator);

    for (size_t i = 0; i < entries_count_list(cfnode->dominates); i++) {
        CFNode* child = read_list(CFNode*, cfnode->dominates)[i];
        visit_domtree(ctx, child, depth + 1);
    }
}

struct List* compute_free_variables(const Scope* scope) {
    struct Dict* ignore_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* free_list = new_list(const Node*);

    Context ctx = {
        .visitor = {
            .visit_fn = (VisitFn) visit_fv,
        },
        .ignore_set = ignore_set,
        .free_list = free_list,
    };

    visit_domtree(&ctx, scope->entry, 0);

    destroy_dict(ignore_set);
    return free_list;
}
