#include "pass.h"

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
        return recreate_node_identity(&ctx->rewriter, old);
    LARRAY(const Node*, elements, width);
    BodyBuilder* bb = begin_block_pure(a);
    Nodes noperands = rewrite_nodes(&ctx->rewriter, old->payload.prim_op.operands);
    for (size_t i = 0; i < width; i++) {
        LARRAY(const Node*, nops, noperands.count);
        for (size_t j = 0; j < noperands.count; j++)
            nops[j] = gen_extract(bb, noperands.nodes[j], singleton(int32_literal(a, i)));
        elements[i] = gen_primop_e(bb, old->payload.prim_op.op, empty(a), nodes(a, noperands.count, nops));
    }
    const Type* t = arr_type(a, (ArrType) {
        .element_type = rewrite_node(&ctx->rewriter, dst_type),
        .size = int32_literal(a, width)
    });
    return yield_values_and_wrap_in_block(bb, singleton(composite_helper(a, t, nodes(a, width, elements))));
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
            if (get_primop_class(node->payload.prim_op.op) & (OcArithmetic | OcLogic | OcCompare | OcShift | OcMath))
                return scalarify_primop(ctx, node);
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_vec_arr(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.validate_builtin_types = false; // TODO: hacky
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
