#include "shady/ir.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    IrArena* dst_arena;
    struct Dict* done;
    struct List* new_fn_vars;
    struct List* new_fn_defs;
} Context;

static const Node* process_block(Context* ctx, const Node* node) {
    assert(node->tag == Block_TAG);
    const Block* old_block = &node->payload.block;

    switch (old_block->terminator->tag) {
        case Callf_TAG: {
            const Callf* old_callf = &old_block->terminator->payload.callf;
            const Node* join;
            // analyse the live stuff in old_join and push so we can recover it
        }
        case Return_TAG: {
            const Return* old_return = &old_block->terminator->payload.fn_ret;
            // pull out the return parameter, place the return args, then push back the return param and return
        }
        default: return node;
    }
}

static const Node* process_function(Context* ctx, const Node* node, bool is_bb) {
    assert(node->tag == Function_TAG);
    const Function* old_fn = &node->payload.fn;
    assert(old_fn->atttributes.is_continuation == is_bb);

    const Node** found = find_value_dict(const Node*, const Node*, ctx->done, node);
    if (found) return *found;

    // if actually a fn ...
    Node* head = fn(ctx->dst_arena, old_fn->atttributes, old_fn->name, old_fn->params, nodes(ctx->dst_arena, 0, NULL));
    insert_dict(const Node*, const Node*, ctx->done, node, head);
    head->payload.fn.block = process_block(ctx, old_fn->block);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* split_callf(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct List* new_fn_vars = new_list(const Node*);
    struct List* new_fn_defs = new_list(const Node*);

    const Root* root = &src_program->payload.root;
    for (size_t i = 0; i < root->declarations.count; i++) {
        // process functions but leave everything else alone
    }

    assert(src_program->tag == Root_TAG);
}
