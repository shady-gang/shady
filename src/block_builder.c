#include "block_builder.h"

#include "list.h"

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

const Node* finish_block(BlockBuilder* builder, const Node* terminator) {
    const Node* fblock = block(builder->arena, (Block) {
        .instructions = nodes(builder->arena, entries_count_list(builder->list), read_list(const Node*, builder->list)),
        .terminator = terminator
    });
    free(builder);
    return fblock;
}