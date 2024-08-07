#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "../ir_private.h"

typedef struct {
    const Type* type;
    size_t size_in_bytes;
    size_t alignment_in_bytes;
} TypeMemLayout;

typedef struct {
    TypeMemLayout mem_layout;
    size_t offset_in_bytes;
} FieldLayout;

TypeMemLayout get_mem_layout(IrArena*, const Type*);

TypeMemLayout get_record_layout(IrArena* a, const Node* record_type, FieldLayout* fields);
size_t get_record_field_offset_in_bytes(IrArena*, const Type*, size_t);

const Type* size_t_type(IrArena* a);
const Node* size_t_literal(IrArena* a, uint64_t value);
const Node* bytes_to_words(BodyBuilder* bb, const Node* bytes);
uint64_t bytes_to_words_static(const IrArena*, uint64_t bytes);
IntSizes float_to_int_width(FloatSizes width);

#endif
