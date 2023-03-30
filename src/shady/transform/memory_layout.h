#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "../ir_private.h"
typedef struct {
    const Type* type;
    size_t size_in_bytes;
} TypeMemLayout;

typedef struct {
    TypeMemLayout mem_layout;
    size_t offset_in_bytes;
} FieldLayout;

TypeMemLayout get_mem_layout(const CompilerConfig*, IrArena*, const Type*);

size_t get_record_layout(const CompilerConfig* config, IrArena* arena, const Node* record_type, FieldLayout* fields);
size_t get_record_field_offset_in_bytes(const CompilerConfig*, IrArena*, const Type*, size_t);

#endif
