#include "passes.h"

#include "../rewrite.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    size_t width;
    const Node* mask;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case QualifiedType_TAG: {
            if (!node->payload.qualified_type.is_uniform) return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = pack_type(arena, (PackType) { .width = ctx->width, .element_type = rewrite_node(&ctx->rewriter, node->payload.qualified_type.type )})
            });
            goto rewrite;
        }
        rewrite:
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void simt2d(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .width = config->subgroup_size,
        .mask = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
