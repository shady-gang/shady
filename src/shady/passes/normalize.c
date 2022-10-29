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
static const Node* force_to_be_value(Context* ctx, const Node* node);

static const Node* rewrite_value(Context* ctx, const Node* node) {
    Context ctx2 = *ctx;
    ctx2.rewriter.rewrite_fn = (RewriteFn) force_to_be_value;
    return rewrite_node(&ctx2.rewriter, node);
}

static const Node* rewrite_something(Context* ctx, const Node* node) {
    Context ctx2 = *ctx;
    ctx2.rewriter.rewrite_fn = (RewriteFn) process_node;
    return rewrite_node(&ctx2.rewriter, node);
}

static const Node* force_to_be_value(Context* ctx, const Node* node) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    const Node* let_bound;
    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            return ref_decl(ctx->rewriter.dst_arena, (RefDecl) { rewrite_something(ctx, node) });
        }
        case Function_TAG: {
            return fn_addr(ctx->rewriter.dst_arena, (FnAddr) { .fn = rewrite_something(ctx, node) });
        }
        // All instructions are let-bound properly
        // TODO: generalize further
        case PrimOp_TAG: {
            assert(ctx->bb);
            let_bound = prim_op(dst_arena, (PrimOp) {
                .op = node->payload.prim_op.op,
                .operands = rewrite_nodes_generic(&ctx->rewriter, (RewriteFn) rewrite_value, node->payload.prim_op.operands),
                .type_arguments = rewrite_nodes_generic(&ctx->rewriter, (RewriteFn) rewrite_something, node->payload.prim_op.type_arguments),
            });
            break;
        }
        case Call_TAG: {
            assert(ctx->bb);
            Nodes oargs = node->payload.call_instr.args;

            const Node* ncallee = rewrite_value(ctx, node->payload.call_instr.callee);

            let_bound = call_instr(dst_arena, (Call) {
                .callee = ncallee,
                .args = rewrite_nodes_generic(&ctx->rewriter, (RewriteFn) rewrite_value, oargs),
            });
            break;
        }
        default: {
            assert(is_value(node));
            const Node* value_or_type = rewrite_something(ctx, node);
            assert(is_value(value_or_type));
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

    // add a builder to each abstraction...
    switch (node->tag) {
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteFn) process_node;

            new->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.anon_lam.body));
            return new;
        }
        case BasicBlock_TAG: {
            Node* new = basic_block(ctx->rewriter.dst_arena, (Node*) rewrite_node(&ctx->rewriter, node->payload.basic_block.fn), recreate_variables(&ctx->rewriter, node->payload.basic_block.params), node->payload.basic_block.name);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, new->payload.basic_block.params);
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteFn) process_node;
            new->payload.basic_block.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.anon_lam.body));
            return new;
        }
        case AnonLambda_TAG: {
            Node* new = lambda(ctx->rewriter.dst_arena, recreate_variables(&ctx->rewriter, node->payload.anon_lam.params));
            register_processed_list(&ctx->rewriter, node->payload.anon_lam.params, new->payload.anon_lam.params);
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_arena);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteFn) process_node;

            new->payload.anon_lam.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.anon_lam.body));
            return new;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void normalize(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .bb = NULL,
    };

    ctx.rewriter.rewrite_field_type.rewrite_value = (RewriteFn) rewrite_value;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
