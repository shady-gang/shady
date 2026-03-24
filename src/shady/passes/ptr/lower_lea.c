#include "shady/passes/ptr_passes.h"

#include "shady/ir/cast.h"
#include "shady/ir/type.h"
#include "shady/analysis/ptr.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

#include "shady/ir/memory_layout.h"

typedef struct {
    Rewriter rewriter;
    const PtrModel* target_mem_model;
    PtrAnalysis* ptr_analysis;
} Context;

static bool is_as_emulated(Context* ctx, AddressSpace as) {
    // if something is not physical in the final target, we need to lower it now
    return !ctx->target_mem_model->address_spaces[as].physical;
}

static const Node* lower_ptr_index(Rewriter* r, const Type* pointed_type, const Node* base, const Node* index) {
    IrArena* a = r->dst_arena;
    size_t base_size = shd_get_type_bitwidth(shd_get_unqualified_type(base->type)) / 8;
    const Type* emulated_ptr_t = int_type(a, (Int) { .width = int_size_from_bytes(base_size), .is_signed = false });

    switch (pointed_type->tag) {
        case VectorType_TAG:
        case ArrType_TAG: {
            const Type* element_type = shd_get_fill_type_element_type(pointed_type);

            const Node* element_t_size = size_of_helper(a, element_type);

            const Node* new_index = shd_convert_int_extend_according_to_src_t(a, emulated_ptr_t, index);
            const Node* physical_offset = prim_op_helper(a, mul_op, mk_nodes(a, new_index, element_t_size));

            return prim_op_helper(a, add_op, mk_nodes(a, base, physical_offset));
        }
        case StructType_TAG: {
            Nodes member_types = pointed_type->payload.struct_type.members;

            const IntLiteral* selector_value = shd_resolve_to_int_literal(index);
            assert(selector_value && "selector value must be known for LEA into a record");
            size_t n = selector_value->value;
            assert(n < member_types.count);

            const Node* offset_of = offset_of_helper(a, pointed_type, shd_uint64_literal(a, n));
            return prim_op_helper(a, add_op, mk_nodes(a, base, offset_of));
        }
        default: shd_error("cannot index into this")
    }
}

static const Node* lower_ptr_offset(Rewriter* r, const Type* pointed_type, const Node* base, const Node* offset) {
    IrArena* a = r->dst_arena;
    size_t base_size = shd_get_type_bitwidth(shd_get_unqualified_type(base->type)) / 8;
    const Type* emulated_ptr_t = int_type(a, (Int) { .width = int_size_from_bytes(base_size), .is_signed = false });

    const Node* ptr = base;

    const IntLiteral* offset_value = shd_resolve_to_int_literal(offset);
    bool offset_is_zero = offset_value && offset_value->value == 0;
    if (!offset_is_zero) {
        // assert(arr_type->tag == ArrType_TAG);
        // const Type* element_type = arr_type->payload.arr_type.element_type;

        const Node* element_t_size = size_of_helper(a, pointed_type);

        const Node* new_offset = shd_convert_int_extend_according_to_src_t(a, emulated_ptr_t, offset);
        const Node* physical_offset = prim_op_helper(a, mul_op, mk_nodes(a, new_offset, element_t_size));

        ptr = prim_op_helper(a, add_op, mk_nodes(a, ptr, physical_offset));
    }

    return ptr;
}

const Node* shd_lower_lea_helper(Rewriter* r, const Node* old) {
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset payload = old->payload.ptr_array_element_offset;
            const Node* old_base = payload.ptr;
            const Type* old_base_ptr_t = old_base->type;
            shd_deconstruct_qualified_type(&old_base_ptr_t);
            assert(old_base_ptr_t->tag == PtrType_TAG);
            const Node* old_result_t = old->type;
            shd_deconstruct_qualified_type(&old_result_t);

            // Nodes new_ops = rewrite_nodes(&ctx->rewriter, old_ops);

            const Node* base = shd_rewrite_node(r, payload.ptr);
            size_t base_size = shd_get_type_bitwidth(shd_get_unqualified_type(base->type)) / 8;
            const Type* emulated_ptr_t = int_type(a, (Int) { .width = int_size_from_bytes(base_size), .is_signed = false });

            const Node* cast_base = bit_cast_helper(a, emulated_ptr_t, base);
            const Type* new_ptr_element_t = shd_rewrite_node(r, shd_get_pointer_type_element(old_base_ptr_t));
            const Node* result = lower_ptr_offset(r, new_ptr_element_t, cast_base, shd_rewrite_node(r, payload.offset));
            const Type* new_ptr_t = shd_rewrite_node(r, old_result_t);
            const Node* cast_result = bit_cast_helper(a, new_ptr_t, result);
            return cast_result;
        }
        case PtrCompositeElement_TAG: {
            PtrCompositeElement payload = old->payload.ptr_composite_element;
            const Node* old_base = payload.ptr;
            const Type* old_base_ptr_t = old_base->type;
            shd_deconstruct_qualified_type(&old_base_ptr_t);
            assert(old_base_ptr_t->tag == PtrType_TAG);
            const Node* old_result_t = old->type;
            shd_deconstruct_qualified_type(&old_result_t);

            const Node* base = shd_rewrite_node(r, payload.ptr);
            size_t base_size = shd_get_type_bitwidth(shd_get_unqualified_type(base->type)) / 8;
            const Type* emulated_ptr_t = int_type(a, (Int) { .width = int_size_from_bytes(base_size), .is_signed = false });

            const Node* cast_base = bit_cast_helper(a, emulated_ptr_t, base);
            const Type* new_ptr_element_t = shd_rewrite_node(r, shd_get_pointer_type_element(old_base_ptr_t));
            const Node* result = lower_ptr_index(r, new_ptr_element_t, cast_base, shd_rewrite_node(r, payload.index));
            const Type* new_ptr_t = shd_rewrite_node(r, old_result_t);
            const Node* cast_result = bit_cast_helper(a, new_ptr_t, result);
            return cast_result;
        }
        default: break;
    }
    shd_error("lower_lea_helper only deals with PtrCompositeElement and PtrArrayElementOffset");
}

static const Node* process(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset lea = old->payload.ptr_array_element_offset;
            const Node* old_base = lea.ptr;
            const Type* old_base_ptr_t = old_base->type;
            shd_deconstruct_qualified_type(&old_base_ptr_t);
            assert(old_base_ptr_t->tag == PtrType_TAG);
            bool must_lower = false;
            must_lower |= !shd_is_logical_memory_declaration(ctx->ptr_analysis, old_base) && is_as_emulated(ctx, old_base_ptr_t->payload.ptr_type.address_space);
            if (!must_lower)
                break;
            return shd_lower_lea_helper(&ctx->rewriter, old);
        }
        case PtrCompositeElement_TAG: {
            PtrCompositeElement lea = old->payload.ptr_composite_element;
            const Node* old_base = lea.ptr;
            const Type* old_base_ptr_t = old_base->type;
            shd_deconstruct_qualified_type(&old_base_ptr_t);
            assert(old_base_ptr_t->tag == PtrType_TAG);
            bool must_lower = false;
            must_lower |= !shd_is_logical_memory_declaration(ctx->ptr_analysis, old_base) && is_as_emulated(ctx, old_base_ptr_t->payload.ptr_type.address_space);
            if (!must_lower)
                break;
            return shd_lower_lea_helper(&ctx->rewriter, old);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_lea(SHADY_UNUSED const CompilerConfig* config, Module* src, const PtrModel* target_mem_model) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    assert(aconfig.rules.ptr.ptr_size == target_mem_model->ptr_size);
    aconfig.optimisations.weaken_bitcast_to_lea = false;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    const UsesMap* uses = shd_new_uses_map_module(src, 0);
    PtrAnalysis* ptr_analysis = shd_new_ptr_analysis(src, uses);

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .target_mem_model = target_mem_model,
        .ptr_analysis = ptr_analysis,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_ptr_analysis(ptr_analysis);
    shd_destroy_uses_map(uses);
    return dst;
}
