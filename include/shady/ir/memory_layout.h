#ifndef SHADY_MEMORY_LAYOUT_H
#define SHADY_MEMORY_LAYOUT_H

#include "shady/ir/base.h"
#include "shady/ir/grammar.h"
#include "shady/config.h"

typedef struct {
    const Type* type;
    size_t size_in_bytes;
    size_t alignment_in_bytes;
} TypeMemLayout;

typedef struct {
    TypeMemLayout mem_layout;
    size_t offset_in_bytes;
} FieldLayout;

TypeMemLayout shd_get_mem_layout(IrArena* a, const Type* type);

TypeMemLayout shd_get_record_layout(IrArena* a, const Node* record_type, FieldLayout* fields);
size_t shd_get_record_field_offset_in_bytes(IrArena* a, const Type* t, size_t i);
size_t shd_get_composite_index_offset_in_bytes(IrArena* a, const Type* t, size_t i);

static inline const Node* size_t_type(IrArena* a) {
    return int_type(a, (Int) { .width = shd_get_arena_config(a)->target.memory.ptr_size, .is_signed = false });
}

static inline const Node* size_t_literal(IrArena* a, uint64_t value) {
    return int_literal(a, (IntLiteral) { .width = shd_get_arena_config(a)->target.memory.ptr_size, .is_signed = false, .value = value });
}

/// Divides the address in bytes to be an address in words. Result type matches that of bytes.
const Node* shd_bytes_to_words(BodyBuilder* bb, const Node* bytes);
uint64_t shd_bytes_to_words_static(const IrArena* a, uint64_t bytes);
ShdIntSize shd_float_to_int_width(ShdFloatFormat width);

size_t shd_get_type_bitwidth(const Type* t);

const Node* _shd_lea_helper(IrArena* a, const Node* ptr, const Node* offset, Nodes indices);
static inline const Node* lea_helper(IrArena* a, const Node* ptr, const Node* offset, Nodes indices) { return _shd_lea_helper(a, ptr, offset, indices); }

#endif
