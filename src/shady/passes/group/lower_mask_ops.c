#include "shady/pass.h"
#include "shady/ir/type.h"
#include "shady/ir/cast.h"

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
        case PrimOp_TAG: {
            Op op = node->payload.prim_op.op;
            Nodes old_nodes = node->payload.prim_op.operands;
            switch(op) {
                case empty_mask_op: return ctx->zero;
                // extract the relevant bit
                case mask_is_thread_active_op: {
                    BodyBuilder* bb = shd_bld_begin_pure(a);
                    const Node* mask = shd_rewrite_node(&ctx->rewriter, old_nodes.nodes[0]);
                    const Node* index = shd_rewrite_node(&ctx->rewriter, old_nodes.nodes[1]);
                    index = shd_bld_conversion(bb, shd_get_exec_mask_type(ctx->rewriter.dst_arena), index);
                    const Node* acc = mask;
                    // acc >>= index
                    acc = prim_op_helper(a, rshift_logical_op, mk_nodes(a, acc, index));
                    // acc &= 0x1
                    acc = prim_op_helper(a, and_op, mk_nodes(a, acc, ctx->one));
                    // acc == 1
                    acc = prim_op_helper(a, eq_op, mk_nodes(a, acc, ctx->one));
                    return shd_bld_to_instr_yield_value(bb, acc);
                }
                default: break;
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lower_mask(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    const Type* mask_type = shd_get_exec_mask_type(a);
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
