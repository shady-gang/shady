#include "passes.h"

#include "../rewrite.h"

typedef struct {
    Rewriter rewriter;
    CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PackType_TAG: {
            return arr_type(a, (ArrType) {
                .element_type = rewrite_node(&ctx->rewriter, node->payload.pack_type.element_type),
                .size = int32_literal(a, node->payload.pack_type.width)
            });
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_vec_arr(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
            .rewriter = create_rewriter(src, dst, (RewriteFn) process),
            .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
