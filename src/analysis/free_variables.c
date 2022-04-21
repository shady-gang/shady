#include "free_variables.h"

#include "../log.h"

#include "dict.h"

#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

struct List* compute_free_variables(Scope* scope) {
    struct Dict* ignore_set = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* free_list = new_list(const Node*);

    #define process(n) { \
        if ((n) && (n)->tag == Variable_TAG && insert_set_get_result(const Node*, ignore_set, n)) \
            append_list(const Node*, free_list, n); \
    }

    for (size_t i = 0; i < scope->size; i++) {
        const CFNode* cfnode = scope->rpo[i];
        assert(cfnode->node->tag == Function_TAG);
        const Function* entry_as_fn = &cfnode->node->payload.fn;

        for (size_t j = 0; j < entry_as_fn->params.count; j++) {
            const Node* param = entry_as_fn->params.nodes[j];
            bool r = insert_set_get_result(const Node*, ignore_set, param);
            assert(r);
        }

        const Block* entry_block = &entry_as_fn->block->payload.block;
        for (size_t j = 0; j < entry_block->instructions.count; j++) {
            const Node* instruction = entry_block->instructions.nodes[j];
            assert(instruction->tag == Let_TAG && "this pass only supports primops currently");

            // ops can be free variables
            Nodes ops = instruction->payload.let.args;
            for (size_t k = 0; k < ops.count; k++) {
                process(ops.nodes[k]);
            }

            // after being computed, outputs are no longer considered free
            Nodes outputs = instruction->payload.let.variables;
            for (size_t k = 0; k < outputs.count; k++) {
                const Node* output = outputs.nodes[k];
                bool r = insert_set_get_result(const Node*, ignore_set, output);
                assert(r);
            }
        }

        switch (entry_block->terminator->tag) {
            case Jump_TAG: {
                const Jump* jp = &entry_block->terminator->payload.jump;
                process(jp->target)
                for (size_t j = 0; j < jp->args.count; j++)
                    process(jp->args.nodes[j]);
                break;
            }
            case Branch_TAG: {
                const Branch* br = &entry_block->terminator->payload.branch;
                process(br->condition)
                process(br->continue_target)
                process(br->merge_target)
                process(br->true_target)
                process(br->false_target)
                for (size_t j = 0; j < br->args.count; j++)
                    process(br->args.nodes[j]);
                break;
            }
            case Return_TAG: {
                const Return* ret = &entry_block->terminator->payload.fn_ret;
                for (size_t j = 0; j < ret->values.count; j++)
                    process(ret->values.nodes[j]);
                break;
            }
            default: error("non-handled terminator");
        }
    }
    destroy_dict(ignore_set);

    return free_list;
}