#include "shady/pass.h"
#include "shady/ir/memory_layout.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "shady/ir.h"

#include <assert.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* guess_pointer_casts(Context* ctx, BodyBuilder* bb, const Node* ptr, const Type* expected_type) {
    IrArena* a = ctx->rewriter.dst_arena;
    while (true) {
        const Type* actual_type = shd_get_unqualified_type(ptr->type);
        assert(actual_type->tag == PtrType_TAG);
        actual_type = shd_get_pointer_type_element(actual_type);
        if (expected_type == actual_type)
            break;

        actual_type = shd_get_maybe_nominal_type_body(actual_type);
        assert(expected_type != actual_type && "todo: rework this function if we change how nominal types are handled");

        switch (actual_type->tag) {
            case RecordType_TAG:
            case ArrType_TAG:
            case PackType_TAG: {
                ptr = lea_helper(a, ptr, shd_int32_literal(a, 0), shd_singleton(shd_int32_literal(a, 0)));
                continue;
            }
            default: break;
        }
        shd_error("Cannot fix pointer")
    }
    return ptr;
}

static const Node* process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    switch (old->tag) {
        case PtrType_TAG: {
            PtrType payload = old->payload.ptr_type;
            if (!shd_get_arena_config(a)->address_spaces[payload.address_space].physical)
                payload.is_reference = true;
            payload.pointed_type = shd_rewrite_node(r, payload.pointed_type);
            return ptr_type(a, payload);
        }
        /*case PtrArrayElementOffset_TAG: {
            Lea payload = old->payload.lea;
            const Type* optr_t = payload.ptr->type;
            deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = rewrite_node(r, optr_t);
            const Node* ptr = rewrite_node(r, payload.ptr);
            const Type* actual_type = get_unqualified_type(ptr->type);
            BodyBuilder* bb = begin_block_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, get_pointer_type_element(expected_type));
            return bind_last_instruction_and_wrap_in_block(bb, lea(a, (Lea) { .ptr = ptr, .offset = rewrite_node(r, payload.offset), .indices = rewrite_nodes(r, payload.indices)}));
        }*/
        // TODO: we actually want to match stuff that has a ptr as an input operand.
        case PtrCompositeElement_TAG: {
            PtrCompositeElement payload = old->payload.ptr_composite_element;
            const Type* optr_t = payload.ptr->type;
            shd_deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = shd_rewrite_node(r, optr_t);
            const Node* ptr = shd_rewrite_node(r, payload.ptr);
            const Type* actual_type = shd_get_unqualified_type(ptr->type);
            BodyBuilder* bb = shd_bld_begin_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, shd_get_pointer_type_element(expected_type));
            return shd_bld_to_instr_with_last_instr(bb, ptr_composite_element(a, (PtrCompositeElement) { .ptr = ptr, .index = shd_rewrite_node(r, payload.index) }));
        }
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op: {
                    const Node* osrc = shd_first(payload.operands);
                    const Type* osrc_t = osrc->type;
                    shd_deconstruct_qualified_type(&osrc_t);
                    if (osrc_t->tag == PtrType_TAG && !shd_get_arena_config(a)->address_spaces[osrc_t->payload.ptr_type.address_space].physical)
                        return shd_rewrite_node(r, osrc);
                    break;
                }
                default: break;
            }
            break;
        }
        case Load_TAG: {
            Load payload = old->payload.load;
            const Type* optr_t = payload.ptr->type;
            shd_deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = shd_rewrite_node(r, optr_t);
            const Node* ptr = shd_rewrite_node(r, payload.ptr);
            const Type* actual_type = shd_get_unqualified_type(ptr->type);
            BodyBuilder* bb = shd_bld_begin_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, shd_get_pointer_type_element(expected_type));
            return load(a, (Load) { .ptr = shd_bld_to_instr_yield_value(bb, ptr), .mem = shd_rewrite_node(r, payload.mem) });
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* optr_t = payload.ptr->type;
            shd_deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = shd_rewrite_node(r, optr_t);
            const Node* ptr = shd_rewrite_node(r, payload.ptr);
            const Type* actual_type = shd_get_unqualified_type(ptr->type);
            BodyBuilder* bb = shd_bld_begin_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, shd_get_pointer_type_element(expected_type));
            return shd_bld_to_instr_with_last_instr(bb, store(a, (Store) { .ptr = ptr, .value = shd_rewrite_node(r, payload.value), .mem = shd_rewrite_node(r, payload.mem) }));
        }
        case GlobalVariable_TAG: {
            AddressSpace as = old->payload.global_variable.address_space;
            if (shd_get_arena_config(a)->address_spaces[as].physical)
                break;
            Nodes annotations = shd_rewrite_nodes(r, old->payload.global_variable.annotations);
            annotations = shd_nodes_append(a, annotations, annotation(a, (Annotation) { .name = "Logical" }));
            Node* new = global_var(ctx->rewriter.dst_module, annotations, shd_rewrite_node(r, old->payload.global_variable.type), old->payload.global_variable.name, as);
            shd_recreate_node_body(r, old, new);
            return new;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

Module* shd_pass_lower_logical_pointers(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.address_spaces[AsInput].physical = false;
    aconfig.address_spaces[AsOutput].physical = false;
    aconfig.address_spaces[AsUniformConstant].physical = false;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
