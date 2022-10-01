#ifndef SHADY_IR_PRIVATE_H
#define SHADY_IR_PRIVATE_H

#include "shady/ir.h"

#include "arena.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    Arena* arena;
    ArenaConfig config;

    VarId next_free_id;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
} IrArena_;

VarId fresh_id(IrArena*);

struct List;
Nodes list_to_nodes(IrArena*, struct List*);

#endif
