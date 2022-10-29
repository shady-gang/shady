#include "passes.h"

#include "../rewrite.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Constant_TAG: return NULL;
        case RefDecl_TAG: {
            if (node->payload.ref_decl.decl->tag == Constant_TAG) {
                return process(ctx, node->payload.ref_decl.decl->payload.constant.value);
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void eliminate_constants(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process)
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
