#include "shady/ir.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../analysis/scope.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    Node* current_fn;
    Node* current_abstraction;
    Scope* current_scope;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    Rewriter * rewriter = &ctx->rewriter;
    IrArena * arena = rewriter->dst_arena;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (is_function(node)) {
        ctx->current_fn = node;
    }

    const Node* restructure = lookup_annotation(ctx->current_fn, "Restructure");
    if (!restructure)
        return recreate_node_identity(&ctx->rewriter, node);

    if (is_function(node)) {
        if (ctx->current_scope) {
            destroy_scope(ctx->current_scope);
            ctx->current_scope = NULL;
        }
    }

    Node* old_abstraction = ctx->current_abstraction;
    if (is_abstraction(node)) {
        ctx->current_abstraction = node;
    }
    Node* result;

    if (node->tag == Branch_TAG) {
        if (!ctx->current_scope)
            ctx->current_scope = new_scope_flipped(ctx->current_fn);

        CFNode* cfnode = scope_lookup(ctx->current_scope, ctx->current_abstraction);
        CFNode* idom = cfnode->idom;

        if(!idom->node) {
            error("Degenerate case: there is no imediate post dominator for this branch.");
        }

        Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

        Nodes yield_types;
        Nodes exit_args;
        Nodes lambda_args;

        Nodes old_params;
        switch (idom->node->tag) {
        case BasicBlock_TAG:
            old_params = idom->node->payload.basic_block.params;
            break;
        case AnonLambda_TAG:
            old_params = idom->node->payload.anon_lam.params;
            break;
        default:
            assert(false);
        }

        if (old_params.count == 0) {
            yield_types = empty(arena);
            exit_args = empty(arena);
            lambda_args = empty(arena);
        } else {
            const Node* types[old_params.count];
            const Node* inner_args[old_params.count];
            const Node* outer_args[old_params.count];

            for (size_t j = 0; j < old_params.count; j++) {
                assert(old_params.nodes[j]->tag == Variable_TAG);
                const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->payload.var.type);

                types[j] = get_unqualified_type(qualified_type);
                inner_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                outer_args[j] = var(arena, NULL, old_params.nodes[j]->payload.var.name);
            }

            yield_types = nodes(arena, old_params.count, types);
            exit_args = nodes(arena, old_params.count, inner_args);
            lambda_args = nodes(arena, old_params.count, outer_args);
        }

        const Node* join_token = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = yield_types
                    }), true), "jp");

        Node* pre_join = basic_block(arena, fn, exit_args, "exit");
        pre_join->payload.basic_block.body = join(arena, (Join) {
                .join_point = join_token,
                .args = exit_args
                });

        const Node* cached = search_processed(rewriter, idom->node);
        if (cached)
            remove_dict(const Node*, rewriter->processed, idom->node);

        register_processed(rewriter, idom->node, pre_join);

        Node* inner_terminator = recreate_node_identity(rewriter, node);

        remove_dict(const Node*, rewriter->processed, idom->node);
        if (cached)
            register_processed(rewriter, idom->node, cached);

        const Node* control_inner = lambda(rewriter->dst_module, singleton(join_token), inner_terminator);
        const Node* new_target = control (arena, (Control) {
                .inside = control_inner,
                .yield_types = yield_types
                });

        const Node* recreated_join = rewrite_node(rewriter, idom->node);

        switch (idom->node->tag) {
        case BasicBlock_TAG: {
            const Node* outer_terminator = jump(arena, (Jump) {
                    .target = recreated_join,
                    .args = lambda_args
                    });

            const Node* anon_lam = lambda(rewriter->dst_module, lambda_args, outer_terminator);
            const Node* empty_let = let(arena, new_target, anon_lam);

            return empty_let;
        }
        case AnonLambda_TAG:
            return let(arena, new_target, recreated_join);
        default:
            assert(false);
        }
    } else {
        result = recreate_node_identity(&ctx->rewriter, node);
    }

    assert(result);

    ctx->current_abstraction = old_abstraction;
    return result;
}

void reconvergence_heuristics(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .current_fn = NULL,
        .current_scope = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
