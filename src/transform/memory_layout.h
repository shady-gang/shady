#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "shady/ir.h"

typedef struct {
    const Type* type;

    size_t size_in_bytes;

    // the size of a cell depends on how physical memory is emulated
    // (what base element type we use in the backing arrays)
    size_t size_in_cells;
} TypeMemLayout;

TypeMemLayout get_mem_layout(const CompilerConfig*, const Type*);

#endif
