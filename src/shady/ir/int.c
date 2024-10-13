#include "shady/ir/int.h"

#include "shady/analysis/literal.h"

#include <assert.h>

const Type* shd_int_type_helper(IrArena* a, bool s, IntSizes w) { return int_type(a, (Int) { .width = w, .is_signed = s }); }

const Type* shd_int8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = true }); }
const Type* shd_int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = true }); }
const Type* shd_int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = true }); }
const Type* shd_int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = true }); }

const Type* shd_uint8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = false }); }
const Type* shd_uint16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = false }); }
const Type* shd_uint32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = false }); }
const Type* shd_uint64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = false }); }

const Node* shd_int8_literal (IrArena* arena, int8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (uint64_t)  (uint8_t) i, .is_signed = true }); }
const Node* shd_int16_literal(IrArena* arena, int16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (uint64_t) (uint16_t) i, .is_signed = true }); }
const Node* shd_int32_literal(IrArena* arena, int32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (uint64_t) (uint32_t) i, .is_signed = true }); }
const Node* shd_int64_literal(IrArena* arena, int64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = (uint64_t) i, .is_signed = true }); }

const Node* shd_uint8_literal (IrArena* arena, uint8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (int64_t) i }); }
const Node* shd_uint16_literal(IrArena* arena, uint16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (int64_t) i }); }
const Node* shd_uint32_literal(IrArena* arena, uint32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (int64_t) i }); }
const Node* shd_uint64_literal(IrArena* arena, uint64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = i }); }

const IntLiteral* shd_resolve_to_int_literal(const Node* node) {
    node = shd_resolve_node_to_definition(node, shd_default_node_resolve_config());
    if (!node)
        return NULL;
    if (node->tag == IntLiteral_TAG)
        return &node->payload.int_literal;
    return NULL;
}

int64_t shd_get_int_literal_value(IntLiteral literal, bool sign_extend) {
    if (sign_extend) {
        switch (literal.width) {
            case IntTy8:  return (int64_t) (int8_t)  (literal.value & 0xFF);
            case IntTy16: return (int64_t) (int16_t) (literal.value & 0xFFFF);
            case IntTy32: return (int64_t) (int32_t) (literal.value & 0xFFFFFFFF);
            case IntTy64: return (int64_t) literal.value;
            default: assert(false);
        }
    } else {
        switch (literal.width) {
            case IntTy8:  return literal.value & 0xFF;
            case IntTy16: return literal.value & 0xFFFF;
            case IntTy32: return literal.value & 0xFFFFFFFF;
            case IntTy64: return literal.value;
            default: assert(false);
        }
    }
}