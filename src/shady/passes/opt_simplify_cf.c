#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"

#include "../analysis/scope.h"
#include "../analysis/callgraph.h"

typedef struct {
    Rewriter rewriter;
    Scope* scope;
    CallGraph* graph;
    Node* fun;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    assert(arena != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
            if (entries_count_dict(fn_node->callers) == 1 && !fn_node->is_address_captured)
                return NULL;

            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            Context fn_ctx = *ctx;
            Scope* scope = new_scope(node);
            fn_ctx.scope = scope;
            fn_ctx.fun = new;
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
            destroy_scope(scope);
            return new;
        }
        case Jump_TAG: {
            const Node* old_tgt = node->payload.jump.target;
            assert(old_tgt && old_tgt->tag == BasicBlock_TAG);
            assert(old_tgt->payload.basic_block.fn == ctx->scope->entry->node);
            CFNode* cfnode = scope_lookup(ctx->scope, old_tgt);
            assert(cfnode);
            size_t preds_count = entries_count_list(cfnode->pred_edges);
            assert(preds_count > 0 && "this CFG looks broken");
            if (preds_count == 1) {
                debugv_print("Inlining jump to %s\n", get_abstraction_name(old_tgt));
                // turn that BB into a lambda !
                Nodes oparams = old_tgt->payload.basic_block.params;
                Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
                Context inline_context = *ctx;
                register_processed_list(&inline_context.rewriter, oparams, nparams);
                const Node* lam = lambda(ctx->rewriter.dst_arena, nparams, rewrite_node(&ctx->rewriter, old_tgt->payload.basic_block.body));
                Nodes args = rewrite_nodes(&ctx->rewriter, node->payload.jump.args);
                return let(arena, quote(arena, args), lam);
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case TailCall_TAG: {
            const Node* dst = node->payload.tail_call.target;
            if (dst->tag == FnAddr_TAG) {
                const Node* dst_fn = dst->payload.fn_addr.fn;
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, dst_fn);
                if (entries_count_dict(fn_node->callers) == 1 && !fn_node->is_address_captured) {
                    debugv_print("Inlining call to %s\n", get_abstraction_name(dst_fn));
                    Nodes oparams = dst_fn->payload.fun.params;
                    Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
                    Context inline_context = *ctx;
                    register_processed_list(&inline_context.rewriter, oparams, nparams);

                    Scope* scope = new_scope(dst_fn);
                    inline_context.scope = scope;
                    const Node* nbody = rewrite_node(&inline_context.rewriter, dst_fn->payload.fun.body);
                    destroy_scope(scope);

                    const Node* lam = lambda(ctx->rewriter.dst_arena, nparams, nbody);
                    Nodes args = rewrite_nodes(&ctx->rewriter, node->payload.tail_call.args);
                    return let(arena, quote(arena, args), lam);
                }
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case BasicBlock_TAG: {
            Nodes params = recreate_variables(&ctx->rewriter, node->payload.basic_block.params);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, params);
            Node* bb = basic_block(arena, (Node*) ctx->fun, params, node->payload.basic_block.name);
            register_processed(&ctx->rewriter, node, bb);
            bb->payload.basic_block.body = process(ctx, node->payload.basic_block.body);
            return bb;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void opt_simplify_cf(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .graph = new_callgraph(src),
        .scope = NULL,
        .fun = NULL,
    };
    rewrite_module(&ctx.rewriter);
    destroy_callgraph(ctx.graph);
    destroy_rewriter(&ctx.rewriter);
}
