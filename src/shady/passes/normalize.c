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
static const Node* force_to_be_type(Context* ctx, const Node* node);

static const Node* rewrite_value(Context* ctx, const Node* node) {
    Context ctx2 = *ctx;
    ctx2.rewriter.rewrite_fn = (RewriteNodeFn) force_to_be_value;
    return ctx2.rewriter.rewrite_fn(&ctx2.rewriter, node);
}

static const Node* rewrite_type(Context* ctx, const Node* node) {
    Context ctx2 = *ctx;
    ctx2.rewriter.rewrite_fn = (RewriteNodeFn) force_to_be_type;
    return ctx2.rewriter.rewrite_fn(&ctx2.rewriter, node);
}

static const Node* rewrite_something(Context* ctx, const Node* node) {
    Context ctx2 = *ctx;
    ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;
    return ctx2.rewriter.rewrite_fn(&ctx2.rewriter, node);
}

static const Node* force_to_be_value(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    IrArena* a = ctx->rewriter.dst_arena;

    const Node* let_bound;
    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            return ref_decl_helper(a, rewrite_something(ctx, node));
        }
        case Function_TAG: {
            return fn_addr_helper(a, rewrite_something(ctx, node));
        }
        case Variable_TAG: return find_processed(&ctx->rewriter, node);
        // All instructions are let-bound properly
        // TODO: generalize further
        case PrimOp_TAG: {
            assert(ctx->bb);
            let_bound = prim_op(a, (PrimOp) {
                .op = node->payload.prim_op.op,
                .operands = rewrite_nodes_with_fn(&ctx->rewriter, node->payload.prim_op.operands, (RewriteNodeFn) rewrite_value),
                .type_arguments = rewrite_nodes_with_fn(&ctx->rewriter, node->payload.prim_op.type_arguments, (RewriteNodeFn) rewrite_something /* TODO: rewire_type ? */),
            });
            break;
        }
        case Call_TAG: {
            assert(ctx->bb);
            Nodes oargs = node->payload.call.args;

            const Node* ncallee = rewrite_value(ctx, node->payload.call.callee);

            let_bound = call(a, (Call) {
                .callee = ncallee,
                .args = rewrite_nodes_with_fn(&ctx->rewriter, oargs, (RewriteNodeFn) rewrite_value),
            });
            break;
        }
        default: {
            assert(is_value(node));
            const Node* value = rewrite_something(ctx, node);
            assert(is_value(value));
            return value;
        }
    }

    return first(bind_instruction_outputs_count(ctx->bb, let_bound, 1, NULL, false));
}

static const Node* force_to_be_type(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;
    switch (node->tag) {
        case NominalType_TAG: {
            return type_decl_ref(ctx->rewriter.dst_arena, (TypeDeclRef) {
                .decl = rewrite_something(ctx, node),
            });
        }
        default: {
            assert(is_type(node));
            const Node* type = rewrite_something(ctx, node);
            assert(is_type(type));
            return type;
        }
    }
}

static const Node* force_to_be_instruction(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (node == NULL) return NULL;

    if (is_instruction(node))
        return rewrite_something(ctx, node);

    const Node* val = force_to_be_value(ctx, node);

    return quote_helper(a, singleton(val));
}

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* a = ctx->rewriter.dst_arena;

    // add a builder to each abstraction...
    switch (node->tag) {
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

            new->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.fun.body));
            return new;
        }
        case BasicBlock_TAG: {
            Node* new = basic_block(a, (Node*) rewrite_node(&ctx->rewriter, node->payload.basic_block.fn), recreate_variables(&ctx->rewriter, node->payload.basic_block.params), node->payload.basic_block.name);
            register_processed(&ctx->rewriter, node, new);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, new->payload.basic_block.params);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;
            new->payload.basic_block.body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.basic_block.body));
            return new;
        }
        case AnonLambda_TAG: {
            Nodes new_params = recreate_variables(&ctx->rewriter, node->payload.anon_lam.params);
            register_processed_list(&ctx->rewriter, node->payload.anon_lam.params, new_params);
            BodyBuilder* bb = begin_body(a);
            Context ctx2 = *ctx;
            ctx2.bb = bb;
            ctx2.rewriter.rewrite_fn = (RewriteNodeFn) process_node;

            const Node* new_body = finish_body(bb, rewrite_node(&ctx2.rewriter, node->payload.anon_lam.body));
            return lambda(a, new_params, new_body);
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* normalize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
        .bb = NULL,
    };

    ctx.rewriter.config.search_map = false;
    ctx.rewriter.config.write_map = false;
    ctx.rewriter.rewrite_field_type.rewrite_value = (RewriteNodeFn) rewrite_value;
    ctx.rewriter.rewrite_field_type.rewrite_type = (RewriteNodeFn) rewrite_type;
    ctx.rewriter.rewrite_field_type.rewrite_instruction = (RewriteNodeFn) force_to_be_instruction;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
