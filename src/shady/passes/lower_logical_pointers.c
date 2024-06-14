#include "pass.h"

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
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

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
        case PrimOp_TAG: {
            PrimOp payload = old->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op: {
                    const Node* osrc = first(payload.operands);
                    const Type* osrc_t = osrc->type;
                    deconstruct_qualified_type(&osrc_t);
                    if (osrc_t->tag == PtrType_TAG && !get_arena_config(a)->address_spaces[osrc_t->payload.ptr_type.address_space].physical)
                        return prim_op_helper(a, quote_op, empty(a), singleton(rewrite_node(r, osrc)));
                    break;
                }
                case load_op:
                case store_op:
                case lea_op: {
                    const Node* optr = first(payload.operands);
                    const Type* optr_t = optr->type;
                    deconstruct_qualified_type(&optr_t);
                    assert(optr_t->tag == PtrType_TAG);
                    const Type* expected_type = rewrite_node(r, optr_t);
                    payload.operands = rewrite_nodes(r, payload.operands);
                    payload.type_arguments = rewrite_nodes(r, payload.type_arguments);
                    const Node* ptr = first(payload.operands);
                    const Type* actual_type = get_unqualified_type(ptr->type);
                    BodyBuilder* bb = begin_body(a);
                    if (expected_type != actual_type)
                        payload.operands = change_node_at_index(a, payload.operands, 0, guess_pointer_casts(ctx, bb, ptr, get_pointer_type_element(expected_type)));
                    return bind_last_instruction_and_wrap_in_block(bb, prim_op(a, payload));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, old);
}

Module* lower_logical_pointers(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.address_spaces[AsInput].physical = false;
    aconfig.address_spaces[AsOutput].physical = false;
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
