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

static const Node* process_operand(Context* ctx, const Node* node) {
    assert(node && is_value(node));

    assert(ctx->bb);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    const Node* let_bound;
    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            Context decl_ctx = *ctx;
                decl_ctx.rewriter.rewrite_fn = (RewriteFn) process_node;
            return ref_decl(ctx->rewriter.dst_arena, (RefDecl) { rewrite_node(&decl_ctx.rewriter, node) });
        }
        case Function_TAG: {
            Context decl_ctx = *ctx;
                decl_ctx.rewriter.rewrite_fn = (RewriteFn) process_node;
            return fn_addr(ctx->rewriter.dst_arena, (FnAddr) { .fn = rewrite_node(&decl_ctx.rewriter, node) });
        }
        // All instructions are let-bound properly
        // TODO: generalize further
        case PrimOp_TAG: {
            Nodes og_ops = node->payload.prim_op.operands;
            LARRAY(const Node*, ops, og_ops.count);
            for (size_t i = 0; i < og_ops.count; i++)
                ops[i] = rewrite_node(&ctx->rewriter, og_ops.nodes[i]);
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
                nargs[i] = rewrite_node(&ctx->rewriter, oargs.nodes[i]);

            Context callee_ctx = *ctx;
            //if (!node->payload.call_instr.is_indirect)
            //    callee_ctx.rewriter.rewrite_fn = (RewriteFn) process_node;
            const Node* ncallee = rewrite_node(&callee_ctx.rewriter, node->payload.call_instr.callee);

            let_bound = call_instr(dst_arena, (Call) {
                .callee = ncallee,
                .args = nodes(dst_arena, oargs.count, nargs)
            });
            break;
        }
        default: {
            assert(is_value(node) || is_type(node));
            const Node* value_or_type = recreate_node_identity(&ctx->rewriter, node);
            assert(is_value(value_or_type) || is_type(value_or_type));
            return value_or_type;
        }
    }

    return bind_instruction_extra(ctx->bb, let_bound, 1, NULL, NULL).nodes[0];
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (is_instruction(node) || (is_terminator(node) && node->tag != Let_TAG)) {
        Context ctx2 = *ctx;
        ctx2.rewriter.rewrite_fn = (RewriteFn) process_operand;
        return recreate_node_identity(&ctx2.rewriter, node);
    }

    switch (node->tag) {
        case Lambda_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteFn) process_node;

            new->payload.lam.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.lam.body));
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
