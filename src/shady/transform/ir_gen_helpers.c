#include "ir_gen_helpers.h"

#include "shady/rewrite.h"
#include "shady/ir/memory_layout.h"

#include "../ir_private.h"

#include "list.h"
#include "portability.h"
#include "log.h"
#include "util.h"

#include <string.h>
#include <assert.h>

Nodes gen_call(BodyBuilder* bb, const Node* callee, Nodes args) {
    assert(shd_get_arena_config(shd_get_bb_arena(bb))->check_types);
    const Node* instruction = call(shd_get_bb_arena(bb), (Call) { .callee = callee, .args = args, .mem = shd_bb_mem(bb) });
    return shd_bld_add_instruction_extract(bb, instruction);
}

const Node* gen_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    return shd_bld_add_instruction(bb, ext_instr(shd_get_bb_arena(bb), (ExtInstr) {
        .mem = shd_bb_mem(bb),
        .set = set,
        .opcode = opcode,
        .result_t = return_t,
        .operands = operands,
    }));
}

const Node* gen_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return prim_op(shd_get_bb_arena(bb), (PrimOp) { .op = reinterpret_op, .operands = shd_singleton(src), .type_arguments = shd_singleton(dst)});
}

const Node* gen_conversion(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return prim_op(shd_get_bb_arena(bb), (PrimOp) { .op = convert_op, .operands = shd_singleton(src), .type_arguments = shd_singleton(dst)});
}

const Node* gen_merge_halves(BodyBuilder* bb, const Node* lo, const Node* hi) {
    IrArena* a = shd_get_bb_arena(bb);
    const Type* src_type = shd_get_unqualified_type(lo->type);
    assert(shd_get_unqualified_type(hi->type) == src_type);
    assert(src_type->tag == Int_TAG);
    IntSizes size = src_type->payload.int_type.width;
    assert(size != IntSizeMax);
    const Type* dst_type = int_type(shd_get_bb_arena(bb), (Int) { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed });
    // widen them
    lo = gen_conversion(bb, dst_type, lo);
    hi = gen_conversion(bb, dst_type, hi);
    // shift hi
    const Node* shift_by = int_literal(shd_get_bb_arena(bb), (IntLiteral)  { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed, .value = shd_get_type_bitwidth(src_type) });
    hi = prim_op_helper(a, lshift_op, shd_empty(a), mk_nodes(a, hi, shift_by));
    // Merge the two
    return prim_op_helper(a, or_op, shd_empty(a), mk_nodes(a, lo, hi));
}

void gen_comment(BodyBuilder* bb, String str) {
    shd_bld_add_instruction_extract(bb, comment(shd_get_bb_arena(bb), (Comment) { .string = str, .mem = shd_bb_mem(bb) }));
}

void gen_debug_printf(BodyBuilder* bb, String pattern, Nodes args) {
    shd_bld_add_instruction(bb, debug_printf(shd_get_bb_arena(bb), (DebugPrintf) { .string = pattern, .args = args, .mem = shd_bb_mem(bb) }));
}

const Node* convert_int_extend_according_to_src_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
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

const Node* convert_int_extend_according_to_dst_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
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

const Node* convert_int_zero_extend(BodyBuilder* bb, const Type* dst_type, const Node* src) {
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

const Node* convert_int_sign_extend(BodyBuilder* bb, const Type* dst_type,  const Node* src) {
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
