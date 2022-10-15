#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "../ir_private.h"

typedef struct {
    const Type* type;
    size_t size_in_bytes;
} TypeMemLayout;

TypeMemLayout get_mem_layout(const CompilerConfig*, IrArena*, const Type*);

static inline size_t bytes_to_i32_cells(size_t size_in_bytes) {
    return (size_in_bytes + 3) / 4;
}

const Node* gen_deserialisation(BodyBuilder*, const Type* element_type, const Node* arr, const Node* base_offset);
void gen_serialisation(BodyBuilder*, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value);

#endif
