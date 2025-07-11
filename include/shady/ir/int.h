#ifndef SHADY_IR_INT_H
#define SHADY_IR_INT_H

#include "shady/ir/grammar.h"
#include "shady/ir/builder.h"

static inline int int_size_in_bytes(ShdIntSize s) {
    switch (s) {
        case ShdIntSize8: return 1;
        case ShdIntSize16: return 2;
        case ShdIntSize32: return 4;
        case ShdIntSize64: return 8;
    }
}

const Type* shd_int_type_helper(IrArena* a, bool s, ShdIntSize w);

const Type* shd_int8_type(IrArena* arena);
const Type* shd_int16_type(IrArena* arena);
const Type* shd_int32_type(IrArena* arena);
const Type* shd_int64_type(IrArena* arena);

const Type* shd_uint8_type(IrArena* arena);
const Type* shd_uint16_type(IrArena* arena);
const Type* shd_uint32_type(IrArena* arena);
const Type* shd_uint64_type(IrArena* arena);

const Node* shd_int8_literal(IrArena* arena, int8_t i);
const Node* shd_int16_literal(IrArena* arena, int16_t i);
const Node* shd_int32_literal(IrArena* arena, int32_t i);
const Node* shd_int64_literal(IrArena* arena, int64_t i);

const Node* shd_uint8_literal(IrArena* arena, uint8_t u);
const Node* shd_uint16_literal(IrArena* arena, uint16_t u);
const Node* shd_uint32_literal(IrArena* arena, uint32_t u);
const Node* shd_uint64_literal(IrArena* arena, uint64_t u);

const IntLiteral* shd_resolve_to_int_literal(const Node* node);
int64_t shd_get_int_literal_value(IntLiteral literal, bool sign_extend);

int64_t shd_get_int_value(const Node* node, bool sign_extend);

const Node* shd_convert_int_extend_according_to_src_t(IrArena*, const Type* dst_type, const Node* src);
const Node* shd_convert_int_extend_according_to_dst_t(IrArena*, const Type* dst_type, const Node* src);
const Node* shd_convert_int_zero_extend(IrArena*, const Type* dst_type, const Node* src);
const Node* shd_convert_int_sign_extend(IrArena*, const Type* dst_type, const Node* src);

#endif
