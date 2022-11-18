#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "../ir_private.h"
typedef struct {
    const Type* type;
    size_t size_in_bytes;
} TypeMemLayout;

TypeMemLayout get_mem_layout(const CompilerConfig*, IrArena*, const Type*);
size_t get_record_field_offset_in_bytes(const CompilerConfig*, IrArena*, const Type*, size_t);

size_t bytes_to_i32_cells(size_t size_in_bytes);

const Node* gen_deserialisation(const CompilerConfig* config, BodyBuilder*, const Type* element_type, const Node* arr, const Node* base_offset);
       void   gen_serialisation(const CompilerConfig* config, BodyBuilder*, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value);

#endif
