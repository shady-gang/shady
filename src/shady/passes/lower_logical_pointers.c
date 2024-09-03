#include "shady/pass.h"

#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

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
        const Type* actual_type = get_unqualified_type(ptr->type);
        assert(actual_type->tag == PtrType_TAG);
        actual_type = get_pointer_type_element(actual_type);
        if (expected_type == actual_type)
            break;

        actual_type = get_maybe_nominal_type_body(actual_type);
        assert(expected_type != actual_type && "todo: rework this function if we change how nominal types are handled");

        switch (actual_type->tag) {
            case RecordType_TAG:
            case ArrType_TAG:
            case PackType_TAG: {
                ptr = gen_lea(bb, ptr, int32_literal(a, 0), singleton(int32_literal(a, 0)));
                continue;
            }
            default: break;
        }
        error("Cannot fix pointer")
    }
    return ptr;
}

static const Node* process(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;

    switch (old->tag) {
        case PtrType_TAG: {
            PtrType payload = old->payload.ptr_type;
            if (!get_arena_config(a)->address_spaces[payload.address_space].physical)
                payload.is_reference = true;
            payload.pointed_type = rewrite_node(r, payload.pointed_type);
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
            deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = rewrite_node(r, optr_t);
            const Node* ptr = rewrite_node(r, payload.ptr);
            const Type* actual_type = get_unqualified_type(ptr->type);
            BodyBuilder* bb = begin_block_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, get_pointer_type_element(expected_type));
            return bind_last_instruction_and_wrap_in_block(bb, ptr_composite_element(a, (PtrCompositeElement) { .ptr = ptr, .index = rewrite_node(r, payload.index)}));
        }
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op: {
                    const Node* osrc = first(payload.operands);
                    const Type* osrc_t = osrc->type;
                    deconstruct_qualified_type(&osrc_t);
                    if (osrc_t->tag == PtrType_TAG && !get_arena_config(a)->address_spaces[osrc_t->payload.ptr_type.address_space].physical)
                        return rewrite_node(r, osrc);
                    break;
                }
                default: break;
            }
            break;
        }
        case Load_TAG: {
            Load payload = old->payload.load;
            const Type* optr_t = payload.ptr->type;
            deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = rewrite_node(r, optr_t);
            const Node* ptr = rewrite_node(r, payload.ptr);
            const Type* actual_type = get_unqualified_type(ptr->type);
            BodyBuilder* bb = begin_block_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, get_pointer_type_element(expected_type));
            return load(a, (Load) { .ptr = yield_value_and_wrap_in_block(bb, ptr), .mem = rewrite_node(r, payload.mem) });
        }
        case Store_TAG: {
            Store payload = old->payload.store;
            const Type* optr_t = payload.ptr->type;
            deconstruct_qualified_type(&optr_t);
            assert(optr_t->tag == PtrType_TAG);
            const Type* expected_type = rewrite_node(r, optr_t);
            const Node* ptr = rewrite_node(r, payload.ptr);
            const Type* actual_type = get_unqualified_type(ptr->type);
            BodyBuilder* bb = begin_block_pure(a);
            if (expected_type != actual_type)
                ptr = guess_pointer_casts(ctx, bb, ptr, get_pointer_type_element(expected_type));
            return bind_last_instruction_and_wrap_in_block(bb, store(a, (Store) { .ptr = ptr, .value = rewrite_node(r, payload.value), .mem = rewrite_node(r, payload.mem) }));
        }
        case GlobalVariable_TAG: {
            AddressSpace as = old->payload.global_variable.address_space;
            if (get_arena_config(a)->address_spaces[as].physical)
                break;
            Nodes annotations = rewrite_nodes(r, old->payload.global_variable.annotations);
            annotations = append_nodes(a, annotations, annotation(a, (Annotation) { .name = "Logical" }));
            Node* new = global_var(ctx->rewriter.dst_module, annotations, rewrite_node(r, old->payload.global_variable.type), old->payload.global_variable.name, as);
            recreate_decl_body_identity(r, old, new);
            return new;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

Module* lower_logical_pointers(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.address_spaces[AsInput].physical = false;
    aconfig.address_spaces[AsOutput].physical = false;
    aconfig.address_spaces[AsUniformConstant].physical = false;
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
