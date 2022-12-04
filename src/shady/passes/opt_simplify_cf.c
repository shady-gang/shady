#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"

#include "../analysis/scope.h"

typedef struct {
    Rewriter rewriter;
    Scope* scope;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            Context fn_ctx = *ctx;
            Scope scope = build_scope(node);
            fn_ctx.scope = &scope;
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
            return new;
        }
        case Jump_TAG: {
            const Node* old_tgt = node->payload.jump.target;
            assert(old_tgt && old_tgt->tag == BasicBlock_TAG);
            CFNode* cfnode = scope_lookup(ctx->scope, old_tgt);
            assert(cfnode);
            size_t preds_count = entries_count_list(cfnode->pred_edges);
            assert(preds_count > 0 && "this CFG looks broken");
            if (preds_count == 1) {
                // turn that BB into a lambda !
                Nodes oparams = old_tgt->payload.basic_block.params;
                Nodes nparams = recreate_variables(&ctx->rewriter, oparams);
                register_processed_list(&ctx->rewriter, oparams, nparams);
                Node* lam = lambda(ctx->rewriter.dst_module, nparams);
                lam->payload.anon_lam.body = rewrite_node(&ctx->rewriter, old_tgt->payload.basic_block.body);
                Nodes args = rewrite_nodes(&ctx->rewriter, node->payload.jump.args);
                const Node* wrapped;
                switch (args.count) {
                    case 0: wrapped = unit(arena); break;
                    case 1: wrapped = quote(arena, first(args)); break;
                    default: wrapped = quote(arena, tuple(arena, args)); break;
                }
                return let(arena, wrapped, lam);
            }
            SHADY_FALLTHROUGH
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void opt_simplify_cf(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .scope = NULL,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
