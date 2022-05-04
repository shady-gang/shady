#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "shady/ir.h"

typedef struct {
    const Type* type;

    size_t size_in_bytes;
} TypeMemLayout;

TypeMemLayout get_mem_layout(const Type*type);

#endif
