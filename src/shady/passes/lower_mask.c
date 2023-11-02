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

    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case MaskType_TAG: return get_actual_mask_type(ctx->rewriter.dst_arena);
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            Nodes old_nodes = node->payload.prim_op.operands;
            switch(op) {
                case empty_mask_op: return quote_helper(a, singleton(ctx->zero));
                case subgroup_active_mask_op: // this is just ballot(true)
                    return prim_op(a, (PrimOp) { .op = subgroup_ballot_op, .type_arguments = empty(a), .operands = singleton(true_lit(ctx->rewriter.dst_arena)) });
                // extract the relevant bit
                case mask_is_thread_active_op: {
                    BodyBuilder* bb = begin_body(a);
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

Module* lower_mask(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    aconfig.specializations.subgroup_mask_representation = SubgroupMaskInt64;
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    const Type* mask_type = get_actual_mask_type(a);
    assert(mask_type->tag == Int_TAG);

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .zero = int_literal(a, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 0 }),
        .one = int_literal(a, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 1 }),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
