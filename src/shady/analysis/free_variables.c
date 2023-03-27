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
        case AnonLambda_TAG:
        case BasicBlock_TAG: break; // we do not visit the insides of functions/basic blocks, that's what the domtree search is already doing!
        default: visit_children(&visitor->visitor, node); break;
    }
}

static void visit_domtree(Context* ctx, CFNode* cfnode, int depth) {
    const Node* abs = cfnode->node;

    bool is_named = abs->tag != AnonLambda_TAG;

    if (is_named) {
        for (int i = 0; i < depth; i++)
            debugvv_print(" ");
        debugvv_print("%s\n", get_abstraction_name(abs));
    }

    // Bind parameters
    Nodes params = get_abstraction_params(abs);
    for (size_t j = 0; j < params.count; j++) {
        const Node* param = params.nodes[j];
        bool r = insert_set_get_result(const Node*, ctx->ignore_set, param);
        // assert(r);
        // this can happen if you visit the domtree of a CFG starting _inside_ a loop
        // we will meet some unbound params but eventually we'll enter their definition after the fact
        // those params should still be considered free in this case.
    }

    const Node* body = get_abstraction_body(abs);
    if (body)
        visit_fv(ctx, body);

    for (size_t i = 0; i < entries_count_list(cfnode->dominates); i++) {
        CFNode* child = read_list(CFNode*, cfnode->dominates)[i];
        visit_domtree(ctx, child, depth + (is_named ? 1 : 0));
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

    debugv_print("Computing free variables...\n");
    visit_domtree(&ctx, scope->entry, 0);

    destroy_dict(ignore_set);
    return free_list;
}
