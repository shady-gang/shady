#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Fill_TAG: {
            const Type* composite_t = rewrite_node(&ctx->rewriter, node->payload.fill.type);
            size_t actual_size = get_int_literal_value(get_fill_type_size(composite_t), false);
            const Node* value = rewrite_node(&ctx->rewriter, node->payload.fill.value);
            LARRAY(const Node*, copies, actual_size);
            for (size_t i = 0; i < actual_size; i++) {
                copies[i] = value;
            }
            return composite(a, composite_t, nodes(a, actual_size, copies));
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

void lower_fill(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
