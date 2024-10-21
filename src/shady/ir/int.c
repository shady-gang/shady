#include "shady/ir/int.h"
#include "shady/ir/type.h"

#include "shady/analysis/literal.h"

#include "log.h"

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

int64_t shd_get_int_value(const Node* node, bool sign_extend) {
    const IntLiteral* lit = shd_resolve_to_int_literal(node);
    if (!lit) shd_error("Not a literal");
    return shd_get_int_literal_value(*lit, sign_extend);
}

const Node* shd_bld_convert_int_extend_according_to_src_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    IrArena* a = shd_get_bb_arena(bb);
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first convert to final bitsize then bitcast
    const Type* extended_src_t = int_type(shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = src_type->payload.int_type.is_signed });
    const Node* val = src;
    val = prim_op_helper(a, convert_op, shd_singleton(extended_src_t), shd_singleton(val));
    val = prim_op_helper(a, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* shd_bld_convert_int_extend_according_to_dst_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    IrArena* a = shd_get_bb_arena(bb);
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first bitcast then convert to final bitsize
    const Type* reinterpreted_src_t = int_type(shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = dst_type->payload.int_type.is_signed });
    const Node* val = src;
    val = prim_op_helper(a, reinterpret_op, shd_singleton(reinterpreted_src_t), shd_singleton(val));
    val = prim_op_helper(a, convert_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* shd_bld_convert_int_zero_extend(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    IrArena* a = shd_get_bb_arena(bb);
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    const Node* val = src;
    val = prim_op_helper(a, reinterpret_op, shd_singleton(
        int_type(shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = false })), shd_singleton(val));
    val = prim_op_helper(a, convert_op, shd_singleton(
        int_type(shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = false })), shd_singleton(val));
    val = prim_op_helper(a, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* shd_bld_convert_int_sign_extend(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    IrArena* a = shd_get_bb_arena(bb);
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    const Node* val = src;
    val = prim_op_helper(a, reinterpret_op, shd_singleton(
        int_type(shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = true })), shd_singleton(val));
    val = prim_op_helper(a, convert_op, shd_singleton(
        int_type(shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = true })), shd_singleton(val));
    val = prim_op_helper(a, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}
