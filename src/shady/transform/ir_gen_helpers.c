#include "ir_gen_helpers.h"

#include "list.h"
#include "portability.h"
#include "log.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"

#include <string.h>
#include <assert.h>

Nodes gen_primop(BodyBuilder* bb, Op op, Nodes type_args, Nodes operands) {
    assert(bb->arena->config.check_types);
    const Node* instruction = prim_op(bb->arena, (PrimOp) { .op = op, .type_arguments = type_args, .operands = operands });
    return bind_instruction(bb, instruction);
}

Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    return gen_primop(bb, op, empty(bb->arena), nodes(bb->arena, operands_count, operands));
}

const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]) {
    Nodes result = gen_primop_c(bb, op, operands_count, operands);
    assert(result.count == 1);
    return result.nodes[0];
}

const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes ty, Nodes nodes) {
    Nodes result = gen_primop(bb, op, ty, nodes);
    return first(result);
}

void gen_push_value_stack(BodyBuilder* bb, const Node* value) {
    gen_primop(bb, push_stack_op, singleton(get_unqualified_type(value->type)), singleton(value));
}

void gen_push_values_stack(BodyBuilder* bb, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(bb, value);
    }
}

const Node* gen_pop_value_stack(BodyBuilder* bb, const Type* type) {
    const Node* instruction = prim_op(bb->arena, (PrimOp) { .op = pop_stack_op, .type_arguments = nodes(bb->arena, 1, (const Node*[]) { type }) });
    return first(bind_instruction(bb, instruction));
}

const Node* gen_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = reinterpret_op, .operands = singleton(src), .type_arguments = singleton(dst)})));
}

const Node* gen_conversion(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = convert_op, .operands = singleton(src), .type_arguments = singleton(dst)})));
}

const Node* gen_merge_halves(BodyBuilder* bb, const Node* lo, const Node* hi) {
    const Type* src_type = get_unqualified_type(lo->type);
    assert(get_unqualified_type(hi->type) == src_type);
    assert(src_type->tag == Int_TAG);
    IntSizes size = src_type->payload.int_type.width;
    assert(size != IntSizeMax);
    const Type* dst_type = int_type(bb->arena, (Int) { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed });
    // widen them
    lo = gen_conversion(bb, dst_type, lo);
    hi = gen_conversion(bb, dst_type, hi);
    // shift hi
    const Node* shift_by = int_literal(bb->arena, (IntLiteral)  { .width = size + 1, .is_signed = src_type->payload.int_type.is_signed, .value.u64 = get_type_bitwidth(src_type) });
    hi = gen_primop_ce(bb, lshift_op, 2, (const Node* []) { hi, shift_by});
    // Merge the two
    return gen_primop_ce(bb, or_op, 2, (const Node* []) { lo, hi });
}

const Node* gen_load(BodyBuilder* bb, const Node* ptr) {
    return gen_primop_ce(bb, load_op, 1, (const Node* []) {ptr });
}

void gen_store(BodyBuilder* instructions, const Node* ptr, const Node* value) {
    gen_primop_c(instructions, store_op, 2, (const Node* []) { ptr, value });
}

const Node* gen_lea(BodyBuilder* bb, const Node* base, const Node* offset, Nodes selectors) {
    LARRAY(const Node*, ops, 2 + selectors.count);
    ops[0] = base;
    ops[1] = offset;
    for (size_t i = 0; i < selectors.count; i++)
        ops[2 + i] = selectors.nodes[i];
    return gen_primop_ce(bb, lea_op, 2 + selectors.count, ops);
}

const Node* gen_extract(BodyBuilder* bb, const Node* base, Nodes selectors) {
    LARRAY(const Node*, ops, 1 + selectors.count);
    ops[0] = base;
    for (size_t i = 0; i < selectors.count; i++)
        ops[1 + i] = selectors.nodes[i];
    return gen_primop_ce(bb, extract_op, 1 + selectors.count, ops);
}

void gen_comment(BodyBuilder* bb, String str) {
    bind_instruction(bb, comment(bb->arena, (Comment) { .string = str }));
}

const Node* find_or_process_decl(Rewriter* rewriter, const char* name) {
    Nodes old_decls = get_module_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (strcmp(get_decl_name(decl), name) == 0) {
            return rewrite_node(rewriter, decl);
        }
    }
    assert(false);
}

const Node* access_decl(Rewriter* rewriter, const char* name) {
    const Node* decl = find_or_process_decl(rewriter, name);
    if (decl->tag == Function_TAG)
        return fn_addr(rewriter->dst_arena, (FnAddr) { .fn = decl });
    else
        return ref_decl(rewriter->dst_arena, (RefDecl) { .decl = decl });
}

const Node* convert_int_extend_according_to_src_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first convert to final bitsize then bitcast
    const Type* extended_src_t = int_type(bb->arena, (Int) { .width = dst_type->payload.int_type.width, .is_signed = src_type->payload.int_type.is_signed });
    const Node* val = src;
    val = gen_primop_e(bb, convert_op, singleton(extended_src_t), singleton(val));
    val = gen_primop_e(bb, reinterpret_op, singleton(dst_type), singleton(val));
    return val;
}

const Node* convert_int_extend_according_to_dst_t(BodyBuilder* bb, const Type* dst_type, const Node* src) {
    const Type* src_type = get_unqualified_type(src->type);
    assert(src_type->tag == Int_TAG);
    assert(dst_type->tag == Int_TAG);

    // first bitcast then convert to final bitsize
    const Type* reinterpreted_src_t = int_type(bb->arena, (Int) { .width = src_type->payload.int_type.width, .is_signed = dst_type->payload.int_type.is_signed });
    const Node* val = src;
    val = gen_primop_e(bb, reinterpret_op, singleton(reinterpreted_src_t), singleton(val));
    val = gen_primop_e(bb, convert_op, singleton(dst_type), singleton(val));
    return val;
}

const Node* get_default_zero_value(IrArena* a, const Type* t) {
    switch (is_type(t)) {
        case NotAType: error("")
        case Type_MaskType_TAG:
        case Type_JoinPointType_TAG: error("TODO");
        case Type_NoRet_TAG: error("Has no values (let alone a default one!)");
        case Type_Int_TAG: return int_literal(a, (IntLiteral) { .width = t->payload.int_type.width, .is_signed = t->payload.int_type.is_signed, .value.u64 = 0 });
        case Type_Float_TAG: return float_literal(a, (FloatLiteral) { .width = t->payload.float_type.width, .value.b64 = 0 });
        case Type_Bool_TAG: return false_lit(a);
        case Type_FnType_TAG:
        case Type_BBType_TAG:
        case Type_LamType_TAG: error("These are symbolic and lack concrete values, so no defaults either.");
        case Type_PtrType_TAG: error("TODO nullptr constant");
        case Type_QualifiedType_TAG: return get_default_zero_value(a, t->payload.qualified_type.type);
        case Type_RecordType_TAG:
        case Type_ArrType_TAG:
        case Type_PackType_TAG:
        case Type_TypeDeclRef_TAG: {
            Nodes elem_tys = get_composite_type_element_types(t);
            LARRAY(const Node*, elems, elem_tys.count);
            for (size_t i = 0; i < elem_tys.count; i++)
                elems[i] = get_default_zero_value(a, elem_tys.nodes[i]);
            return composite(a, t, nodes(a, elem_tys.count, elems));
        }
    }
}
