#include "free_variables.h"

#include "../log.h"
#include "../visit.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct VisitorFV_ {
    Visitor visitor;
    struct Dict* ignore_set;
    struct List* free_list;
} VisitorFV;

static void visit_fv(VisitorFV* visitor, const Node* node) {
    assert(node);
    switch (node->tag) {
        case Variable_TAG: {
            // if we encounter a node we haven't ignored already, it is deemed free
            if (insert_set_get_result(const Node*, visitor->ignore_set, node))
                append_list(const Node*, visitor->free_list, node);
            break;
        }
        case Function_TAG: {
            const Function* fun = &node->payload.fn;

            // Bind parameters
            for (size_t j = 0; j < fun->params.count; j++) {
                const Node* param = fun->params.nodes[j];
                bool r = insert_set_get_result(const Node*, visitor->ignore_set, param);
                assert(r);
            }

            const Block* entry_block = &fun->block->payload.block;
            assert(fun->block);
            for (size_t j = 0; j < entry_block->instructions.count; j++) {
                const Node* let_node = entry_block->instructions.nodes[j];
                assert(let_node->tag == Let_TAG);

                visit_fv(visitor, let_node->payload.let.instruction);

                // after being computed, outputs are no longer considered free
                Nodes outputs = let_node->payload.let.variables;
                for (size_t k = 0; k < outputs.count; k++) {
                    const Node* output = outputs.nodes[k];
                    bool r = insert_set_get_result(const Node*, visitor->ignore_set, output);
                    assert(r);
                }
            }

            visit_fv(visitor, entry_block->terminator);

            if (!fun->is_basic_block)
                visit_fn_blocks_except_head(&visitor->visitor, node);
            break;
        }
        case Block_TAG:
        case Root_TAG: error("should not be reachable")
        default: visit_children(&visitor->visitor, node); break;
    }
}

struct List* compute_free_variables(const Node* entry) {
    struct Dict* ignore_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* free_list = new_list(const Node*);

    assert(entry && entry->tag == Function_TAG);

    VisitorFV visitor_fv = {
        .visitor = {
            .visit_fn = (VisitFn) visit_fv,
            .visit_fn_scope_rpo = true,
            .visit_cf_targets = false,
            .visit_return_fn_annotation = false,
        },
        .ignore_set = ignore_set,
        .free_list = free_list,
    };

    visit_fv(&visitor_fv, entry);

    destroy_dict(ignore_set);
    return free_list;
}
