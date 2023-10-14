#include "l2s_private.h"

#include "portability.h"
#include "../shady/rewrite.h"
#include "../shady/type.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Variable_TAG: return var(a, qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.var.type), false), node->payload.var.name);
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

void postprocess(Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
    };

    ctx.rewriter.config.process_variables = true;
    // ctx.rewriter.config.search_map = false;
    // ctx.rewriter.config.write_map = false;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
