#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    BodyBuilder* bb;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* ensure_is_value(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    //const Node* already_done = search_processed(&ctx->rewriter, node);
    //if (already_done)
    //    return already_done;

    assert(ctx->bb);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    const Node* let_bound;
    switch (node->tag) {
        case PrimOp_TAG: {
            Nodes og_ops = node->payload.prim_op.operands;
            LARRAY(const Node*, ops, og_ops.count);
            for (size_t i = 0; i < og_ops.count; i++)
                ops[i] = ensure_is_value(ctx, og_ops.nodes[i]);
            let_bound = prim_op(dst_arena, (PrimOp) {
                .op = node->payload.prim_op.op,
                .operands = nodes(dst_arena, og_ops.count, ops)
            });
            break;
        }
        case Call_TAG: {
            Nodes oargs = node->payload.call_instr.args;
            LARRAY(const Node*, nargs, oargs.count);
            for (size_t i = 0; i < oargs.count; i++)
                nargs[i] = ensure_is_value(ctx, oargs.nodes[i]);
            const Node* ncallee = node->payload.call_instr.is_indirect ? ensure_is_value(ctx, node->payload.call_instr.callee) : process_node(ctx, node->payload.call_instr.callee);
            let_bound = call_instr(dst_arena, (Call) {
                .is_indirect = node->payload.call_instr.is_indirect,
                .callee = ncallee,
                .args = nodes(dst_arena, oargs.count, nargs)
            });
            break;
        }
        default: return process_node(ctx, node);
    }

    return bind_instruction_extra(ctx->bb, let_bound, 1, NULL, NULL).nodes[0];
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (is_instruction(node) || (is_terminator(node) && node->tag != Let_TAG)) {
        Context must_be_values_ctx = *ctx;
        must_be_values_ctx.rewriter.rewrite_fn = (RewriteFn) ensure_is_value;
        return recreate_node_identity(&must_be_values_ctx.rewriter, node);
    }

    switch (node->tag) {
        case Lambda_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
            Context in_bb_ctx = *ctx;
            in_bb_ctx.bb = bb;
            in_bb_ctx.rewriter.rewrite_fn = (RewriteFn) process_node;

            new->payload.lam.body = finish_body(bb, rewrite_node(&in_bb_ctx.rewriter, node->payload.lam.body));
            return new;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void normalize(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .bb = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
