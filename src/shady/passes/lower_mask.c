#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    const Node* zero;
    const Node* one;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case MaskType_TAG: return get_actual_mask_type(ctx->rewriter.dst_arena);
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            Nodes old_nodes = node->payload.prim_op.operands;
            switch(op) {
                case empty_mask_op: return ctx->zero;
                // extract the relevant bit
                case mask_is_thread_active_op: {
                    BodyBuilder* bb = begin_block_pure(a);
                    const Node* mask = shd_rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                    const Node* index = shd_rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                    index = gen_conversion(bb, get_actual_mask_type(ctx->rewriter.dst_arena), index);
                    const Node* acc = mask;
                    // acc >>= index
                    acc = gen_primop_ce(bb, rshift_logical_op, 2, (const Node* []) { acc, index });
                    // acc &= 0x1
                    acc = gen_primop_ce(bb, and_op, 2, (const Node* []) { acc, ctx->one });
                    // acc == 1
                    acc = gen_primop_ce(bb, eq_op, 2, (const Node* []) { acc, ctx->one });
                    return yield_value_and_wrap_in_block(bb, acc);
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* lower_mask(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    aconfig.specializations.subgroup_mask_representation = SubgroupMaskInt64;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    const Type* mask_type = get_actual_mask_type(a);
    assert(mask_type->tag == Int_TAG);

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .zero = int_literal(a, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 0 }),
        .one = int_literal(a, (IntLiteral) { .width = mask_type->payload.int_type.width, .value = 1 }),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
