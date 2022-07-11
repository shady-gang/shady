#include "block_builder.h"
#include "rewrite.h"
#include "fold.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>

BlockBuilder* begin_block(IrArena* arena) {
    BlockBuilder* builder = malloc(sizeof(BlockBuilder));
    *builder = (BlockBuilder) {
        .arena = arena,
        .list = new_list(const Node*)
    };
    return builder;
}

Nodes append_block(BlockBuilder* builder, const Node* instruction) {
    append_list(const Node*, builder->list, instruction);
}

void copy_instrs(BlockBuilder* builder, Nodes instructions) {
    for (size_t i = 0; i < instructions.count; i++)
        append_block(builder, instructions.nodes[i]);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    if (is_instruction(node) || is_terminator(node))
        return recreate_node_identity(&ctx->rewriter, node);
    if (node->tag == Variable_TAG) {
        const Node* found = search_processed(&ctx->rewriter, node);
        if (found) node = found;
    }
    return resolve_known_vars(node, true);
    return node;
}

const Node* finish_block(BlockBuilder* builder, const Node* terminator) {
    struct List* final_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context ctx = {
        .rewriter = {
            .dst_arena = builder->arena,
            .src_arena = builder->arena,
            .rewrite_fn = (RewriteFn) process_node,
            .rewrite_decl_body = NULL,
            .processed = done,
        },
    };

    for (size_t i = 0; i < entries_count_list(builder->list); i++) {
        const Node* instruction = read_list(const Node*, builder->list)[i];
        instruction = process_node(&ctx, instruction);
        append_list(const Node*, final_list, instruction);
    }

    destroy_list(builder->list);

    const Node* fblock = block(builder->arena, (Block) {
        .instructions = nodes(builder->arena, entries_count_list(final_list), read_list(const Node*, final_list)),
        .terminator = process_node(&ctx, terminator)
    });

    destroy_dict(done);
    destroy_list(final_list);
    free(builder);
    return fblock;
}