#ifndef SHADY_ARENA_H
#define SHADY_ARENA_H

#include "shady/ir.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;

    IrConfig config;

    VarId next_free_id;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
} IrArena_;

void* arena_alloc(IrArena* arena, size_t size);
VarId fresh_id(IrArena*);

struct List;
Nodes list_to_nodes(IrArena*, struct List*);

#endif
