#ifndef SHADY_BLOCK_BUILDER_H
#define SHADY_BLOCK_BUILDER_H

#include "shady/ir.h"

struct BlockBuilder_ {
    IrArena* arena;
    struct List* list;
};

#endif
