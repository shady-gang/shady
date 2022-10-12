#include "shady/ir.h"

#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "list.h"

#include "dict.h"

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
            let_bound = let(dst_arena, prim_op(dst_arena, (PrimOp) {
                .op = node->payload.prim_op.op,
                .operands = nodes(dst_arena, og_ops.count, ops)
            }), 1, NULL);
            break;
        }
        case Call_TAG: {
            Nodes oargs = node->payload.call_instr.args;
            LARRAY(const Node*, nargs, oargs.count);
            for (size_t i = 0; i < oargs.count; i++)
                nargs[i] = ensure_is_value(ctx, oargs.nodes[i]);
            const Node* ncallee = node->payload.call_instr.is_indirect ? ensure_is_value(ctx, node->payload.call_instr.callee) : process_node(ctx, node->payload.call_instr.callee);
            let_bound = let(dst_arena, call_instr(dst_arena, (Call) {
                .is_indirect = node->payload.call_instr.is_indirect,
                .callee = ncallee,
                .args = nodes(dst_arena, oargs.count, nargs)
            }), 1, NULL);
            break;
        }
        default: {
            return process_node(ctx, node);
        }
    }

    append_body(ctx->bb, let_bound);
    //register_processed(&ctx->rewriter, node, let_bound->payload.let.variables.nodes[0]);
    return let_bound->payload.let.variables.nodes[0];
}

static const Node* process_body(Context* ctx, const Node* old_body) {
    BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
    Context in_bb_ctx = *ctx;
    in_bb_ctx.rewriter.rewrite_fn = (RewriteFn) ensure_is_value;
    in_bb_ctx.bb = bb;

    Nodes old_instructions = old_body->payload.body.instructions;
    for (size_t i = 0; i < old_instructions.count; i++)
        append_body(bb, recreate_node_identity(&in_bb_ctx.rewriter, old_instructions.nodes[i]));

    return finish_body(bb, recreate_node_identity(&in_bb_ctx.rewriter, old_body->payload.body.terminator));
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    switch (node->tag) {
        case Body_TAG: return process_body(ctx, node);
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* normalize(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) process_node,
            .processed = done,
        },
        .bb = NULL,
    };

    assert(src_program->tag == Root_TAG);

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);

    destroy_dict(done);
    return rewritten;
}
