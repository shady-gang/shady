#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    const Node* zero;
    const Node* one;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case MaskType_TAG: return get_actual_mask_type(ctx->rewriter.dst_arena);
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            Nodes old_nodes = node->payload.prim_op.operands;
            switch(op) {
                case empty_mask_op: return quote_single(arena, ctx->zero);
                case subgroup_active_mask_op: // this is just ballot(true)
                    return prim_op(arena, (PrimOp) { .op = subgroup_ballot_op, .type_arguments = empty(arena), .operands = singleton(true_lit(ctx->rewriter.dst_arena)) });
                // extract the relevant bit
                case mask_is_thread_active_op: {
                    BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                    const Node* mask = rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                    const Node* index = rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                    index = gen_conversion(bb, get_actual_mask_type(ctx->rewriter.dst_arena), index);
                    const Node* acc = mask;
                    // acc >>= index
                    acc = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []) { acc, index });
                    // acc &= 0x1
                    acc = gen_primop_ce(bb, and_op, 2, (const Node* []) { acc, ctx->one });
                    // acc == 1
                    acc = gen_primop_ce(bb, eq_op, 2, (const Node* []) { acc, ctx->one });
                    return yield_values_and_wrap_in_block(bb, singleton(acc));
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_mask(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    IrArena* arena = dst->arena;
    const Type* mask_type = get_actual_mask_type(arena);
    assert(mask_type->tag == Int_TAG);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .zero = int_literal(arena, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 0 }),
        .one = int_literal(arena, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 1 }),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
