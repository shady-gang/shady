#include "passes.h"

#include "dict.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"

#include "../analysis/callgraph.h"

typedef struct {
    Rewriter rewriter;
    CallGraph* graph;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            if (!fn_node->is_address_captured && !fn_node->is_recursive && !fn_node->has_indirect_call) {
                annotations = append_nodes(arena, annotations, annotation(arena, (Annotation) {
                    .name = "Leaf",
                }));
            }
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            recreate_decl_body_identity(&ctx->rewriter, node, new);
            return new;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void mark_leaf_functions(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .graph = get_callgraph(src)
    };
    rewrite_module(&ctx.rewriter);
    dispose_callgraph(ctx.graph);
    destroy_rewriter(&ctx.rewriter);
}
