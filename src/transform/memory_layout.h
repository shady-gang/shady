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

TypeMemLayout get_mem_layout(const CompilerConfig*, IrArena*, const Type*);

const Node* gen_deserialisation(BlockBuilder*, const Type* element_type, const Node* arr, const Node* base_offset);
void gen_serialisation(BlockBuilder*, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value);

#endif
