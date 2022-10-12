#ifndef SHADY_BODY_BUILDER_H
#define SHADY_BODY_BUILDER_H

#include "shady/ir.h"

struct BodyBuilder_ {
    IrArena* arena;
    struct List* list;
};

#endif
