#include "shady/pass.h"

#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* scalarify_primop(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* dst_type = old->type;
    deconstruct_qualified_type(&dst_type);
    size_t width = deconstruct_maybe_packed_type(&dst_type);
    if (width == 1)
        return shd_recreate_node(&ctx->rewriter, old);
    LARRAY(const Node*, elements, width);
    BodyBuilder* bb = begin_block_pure(a);
    Nodes noperands = shd_rewrite_nodes(&ctx->rewriter, old->payload.prim_op.operands);
    for (size_t i = 0; i < width; i++) {
        LARRAY(const Node*, nops, noperands.count);
        for (size_t j = 0; j < noperands.count; j++)
            nops[j] = gen_extract(bb, noperands.nodes[j], shd_singleton(shd_int32_literal(a, i)));
        elements[i] = gen_primop_e(bb, old->payload.prim_op.op, shd_empty(a), shd_nodes(a, noperands.count, nops));
    }
    const Type* t = arr_type(a, (ArrType) {
        .element_type = shd_rewrite_node(&ctx->rewriter, dst_type),
        .size = shd_int32_literal(a, width)
    });
    return yield_values_and_wrap_in_block(bb, shd_singleton(composite_helper(a, t, shd_nodes(a, width, elements))));
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PackType_TAG: {
            return arr_type(a, (ArrType) {
                .element_type = shd_rewrite_node(&ctx->rewriter, node->payload.pack_type.element_type),
                .size = shd_int32_literal(a, node->payload.pack_type.width)
            });
        }
        case PrimOp_TAG: {
            if (shd_get_primop_class(node->payload.prim_op.op) & (OcArithmetic | OcLogic | OcCompare | OcShift | OcMath))
                return scalarify_primop(ctx, node);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* lower_vec_arr(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.validate_builtin_types = false; // TODO: hacky
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
