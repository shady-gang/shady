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
    assert(shd_get_arena_config(_shd_get_bb_arena(bb))->check_types);
    const Node* instruction = call(_shd_get_bb_arena(bb), (Call) { .callee = callee, .args = args, .mem = shd_bb_mem(bb) });
    return shd_bld_add_instruction_extract(bb, instruction);
}

Nodes gen_primop(BodyBuilder* bb, Op op, Nodes type_args, Nodes operands) {
    assert(shd_get_arena_config(_shd_get_bb_arena(bb))->check_types);
    const Node* instruction = prim_op(_shd_get_bb_arena(bb), (PrimOp) { .op = op, .type_arguments = type_args, .operands = operands });
    return shd_singleton(instruction);
}

Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    return gen_primop(bb, op, shd_empty(_shd_get_bb_arena(bb)), shd_nodes(_shd_get_bb_arena(bb), operands_count, operands));
}

const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    Nodes result = gen_primop_c(bb, op, operands_count, operands);
    assert(result.count == 1);
    return result.nodes[0];
}

const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes ty, Nodes nodes) {
    Nodes result = gen_primop(bb, op, ty, nodes);
    return shd_first(result);
}

const Node* gen_ext_instruction(BodyBuilder* bb, String set, int opcode, const Type* return_t, Nodes operands) {
    return shd_bld_add_instruction(bb, ext_instr(_shd_get_bb_arena(bb), (ExtInstr) {
        .mem = shd_bb_mem(bb),
        .set = set,
        .opcode = opcode,
        .result_t = return_t,
        .operands = operands,
    }));
}

void gen_push_value_stack(BodyBuilder* bb, const Node* value) {
    shd_bld_add_instruction_extract(bb, push_stack(_shd_get_bb_arena(bb), (PushStack) { .value = value, .mem = shd_bb_mem(bb) }));
}

void gen_push_values_stack(BodyBuilder* bb, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(bb, value);
    }
}

const Node* gen_pop_value_stack(BodyBuilder* bb, const Type* type) {
    const Node* instruction = pop_stack(_shd_get_bb_arena(bb), (PopStack) { .type = type, .mem = shd_bb_mem(bb) });
    return shd_first(shd_bld_add_instruction_extract(bb, instruction));
}

const Node* gen_get_stack_base_addr(BodyBuilder* bb) {
    return get_stack_base_addr(_shd_get_bb_arena(bb), (GetStackBaseAddr) { .mem = shd_bb_mem(bb) });
}

const Node* gen_get_stack_size(BodyBuilder* bb) {
    return shd_first(shd_bld_add_instruction_extract(bb, get_stack_size(_shd_get_bb_arena(bb), (GetStackSize) { .mem = shd_bb_mem(bb) })));
}

void gen_set_stack_size(BodyBuilder* bb, const Node* new_size) {
    shd_bld_add_instruction_extract(bb, set_stack_size(_shd_get_bb_arena(bb), (SetStackSize) { .value = new_size, .mem = shd_bb_mem(bb) }));
}

const Node* gen_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return prim_op(_shd_get_bb_arena(bb), (PrimOp) { .op = reinterpret_op, .operands = shd_singleton(src), .type_arguments = shd_singleton(dst)});
}

const Node* gen_conversion(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return prim_op(_shd_get_bb_arena(bb), (PrimOp) { .op = convert_op, .operands = shd_singleton(src), .type_arguments = shd_singleton(dst)});
}

const Node* gen_merge_halves(BodyBuilder* bb, const Node* lo, const Node* hi) {
    const Type* src_type = shd_get_unqualified_type(lo->type);
    assert(shd_get_unqualified_type(hi->type) == src_type);
    assert(src_type->tag == Int_TAG);
    IntSizes size = src_type->payload.int_type.width;
    assert(size != IntSizeMax);
    const Type* dst_type = int_type(_shd_get_bb_arena(bb), (Int) { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed });
    // widen them
    lo = gen_conversion(bb, dst_type, lo);
    hi = gen_conversion(bb, dst_type, hi);
    // shift hi
    const Node* shift_by = int_literal(_shd_get_bb_arena(bb), (IntLiteral)  { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed, .value = shd_get_type_bitwidth(src_type) });
    hi = gen_primop_ce(bb, lshift_op, 2, (const Node* []) { hi, shift_by});
    // Merge the two
    return gen_primop_ce(bb, or_op, 2, (const Node* []) { lo, hi });
}

const Node* gen_stack_alloc(BodyBuilder* bb, const Type* type) {
    return shd_first(shd_bld_add_instruction_extract(bb, stack_alloc(_shd_get_bb_arena(bb), (StackAlloc) { .type = type, .mem = shd_bb_mem(bb) })));
}

const Node* gen_local_alloc(BodyBuilder* bb, const Type* type) {
    return shd_first(shd_bld_add_instruction_extract(bb, local_alloc(_shd_get_bb_arena(bb), (LocalAlloc) { .type = type, .mem = shd_bb_mem(bb) })));
}

const Node* gen_load(BodyBuilder* bb, const Node* ptr) {
    return shd_first(shd_bld_add_instruction_extract(bb, load(_shd_get_bb_arena(bb), (Load) { .ptr = ptr, .mem = shd_bb_mem(bb) })));
}

void gen_store(BodyBuilder* bb, const Node* ptr, const Node* value) {
    shd_bld_add_instruction_extract(bb, store(_shd_get_bb_arena(bb), (Store) { .ptr = ptr, .value = value, .mem = shd_bb_mem(bb) }));
}

const Node* gen_lea(BodyBuilder* bb, const Node* base, const Node* offset, Nodes selectors) {
    return lea_helper(_shd_get_bb_arena(bb), base, offset, selectors);
}

void gen_comment(BodyBuilder* bb, String str) {
    shd_bld_add_instruction_extract(bb, comment(_shd_get_bb_arena(bb), (Comment) { .string = str, .mem = shd_bb_mem(bb) }));
}

void gen_debug_printf(BodyBuilder* bb, String pattern, Nodes args) {
    shd_bld_add_instruction(bb, debug_printf(_shd_get_bb_arena(bb), (DebugPrintf) { .string = pattern, .args = args, .mem = shd_bb_mem(bb) }));
}

const Node* get_builtin(Module* m, Builtin b) {
    Nodes decls = shd_module_get_declarations(m);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG)
            continue;
        const Node* a = shd_lookup_annotation(decl, "Builtin");
        if (!a)
            continue;
        String builtin_name = shd_get_annotation_string_payload(a);
        assert(builtin_name);
        if (strcmp(builtin_name, shd_get_builtin_name(b)) == 0)
            return decl;
    }

    return NULL;
}

const Node* get_or_create_builtin(Module* m, Builtin b, String n) {
    const Node* decl = get_builtin(m, b);
    if (decl)
        return decl;

    AddressSpace as = shd_get_builtin_address_space(b);
    IrArena* a = shd_module_get_arena(m);
    decl = global_var(m, shd_singleton(annotation_value_helper(a, "Builtin", string_lit_helper(a,
                                                                                               shd_get_builtin_name(b)))),
                      shd_get_builtin_type(a, b), n ? n : shd_format_string_arena(a->arena, "builtin_%s",
                                                                                                                                                                                       shd_get_builtin_name(
                                                                                                                                                                                          b)), as);
    return decl;
}

const Node* gen_builtin_load(Module* m, BodyBuilder* bb, Builtin b) {
    return gen_load(bb, ref_decl_helper(_shd_get_bb_arena(bb), get_or_create_builtin(m, b, NULL)));
}

const Node* find_or_process_decl(Rewriter* rewriter, const char* name) {
    Nodes old_decls = shd_module_get_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (strcmp(get_declaration_name(decl), name) == 0) {
            return shd_rewrite_node(rewriter, decl);
        }
    }
    assert(false);
}

const Node* access_decl(Rewriter* rewriter, const char* name) {
    const Node* decl = find_or_process_decl(rewriter, name);
    if (decl->tag == Function_TAG)
        return fn_addr_helper(rewriter->dst_arena, decl);
    else
        return ref_decl_helper(rewriter->dst_arena, decl);
}

const Node* convert_int_extend_according_to_src_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first convert to final bitsize then bitcast
    const Type* extended_src_t = int_type(_shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = src_type->payload.int_type.is_signed });
    const Node* val = src;
    val = gen_primop_e(bb, convert_op, shd_singleton(extended_src_t), shd_singleton(val));
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* convert_int_extend_according_to_dst_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first bitcast then convert to final bitsize
    const Type* reinterpreted_src_t = int_type(_shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = dst_type->payload.int_type.is_signed });
    const Node* val = src;
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(reinterpreted_src_t), shd_singleton(val));
    val = gen_primop_e(bb, convert_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* convert_int_zero_extend(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    const Node* val = src;
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(
        int_type(_shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = false })), shd_singleton(val));
    val = gen_primop_e(bb, convert_op, shd_singleton(
        int_type(_shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = false })), shd_singleton(val));
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}

const Node* convert_int_sign_extend(BodyBuilder* bb, const Type* dst_type,  const Node* src) {
    const Type* src_type = shd_get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    const Node* val = src;
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(
        int_type(_shd_get_bb_arena(bb), (Int) { .width = src_type->payload.int_type.width, .is_signed = true })), shd_singleton(val));
    val = gen_primop_e(bb, convert_op, shd_singleton(
        int_type(_shd_get_bb_arena(bb), (Int) { .width = dst_type->payload.int_type.width, .is_signed = true })), shd_singleton(val));
    val = gen_primop_e(bb, reinterpret_op, shd_singleton(dst_type), shd_singleton(val));
    return val;
}
