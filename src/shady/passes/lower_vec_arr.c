#include "passes.h"

#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* scalarify_primop(Context* ctx, const Node* old) {
    IrArena* a = ctx->rewriter.dst_arena;
    const Type* dst_type = old->type;
    deconstruct_qualified_type(&dst_type);
    size_t width = deconstruct_maybe_packed_type(&dst_type);
    if (width == 1)
        return recreate_node_identity(&ctx->rewriter, old);
    LARRAY(const Node*, elements, width);
    BodyBuilder* bb = begin_body(a);
    Nodes noperands = rewrite_nodes(&ctx->rewriter, old->payload.prim_op.operands);
    for (size_t i = 0; i < width; i++) {
        LARRAY(const Node*, nops, noperands.count);
        for (size_t j = 0; j < noperands.count; j++)
            nops[j] = gen_extract(bb, noperands.nodes[j], singleton(int32_literal(a, j)));
        elements[i] = gen_primop_e(bb, old->payload.prim_op.op, empty(a), nodes(a, noperands.count, nops));
    }
    const Type* t = arr_type(a, (ArrType) {
        .element_type = rewrite_node(&ctx->rewriter, dst_type),
        .size = int32_literal(a, width)
    });
    return yield_values_and_wrap_in_block(bb, singleton(composite(a, t, nodes(a, width, elements))));
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PackType_TAG: {
            return arr_type(a, (ArrType) {
                .element_type = rewrite_node(&ctx->rewriter, node->payload.pack_type.element_type),
                .size = int32_literal(a, node->payload.pack_type.width)
            });
        }
        case PrimOp_TAG: {
#define HANDLE(_, o) case o##_op: return scalarify_primop(ctx, node);
            switch (node->payload.prim_op.op) {
                ARITHM_PRIMOPS(HANDLE)
                BITSTUFF_PRIMOPS(HANDLE)
                CMP_PRIMOPS(HANDLE)
                SHIFT_PRIMOPS(HANDLE)
                default: break;
            }
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_vec_arr(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
