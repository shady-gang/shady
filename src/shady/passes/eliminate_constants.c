#include "shady/ir.h"

#include "../rewrite.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Root_TAG: {
            Nodes old_decls = node->payload.root.declarations;
            size_t new_decls_count = 0;
            LARRAY(const Node*, decls, old_decls.count);
            for (size_t i = 0; i < old_decls.count; i++) {
                if (old_decls.nodes[i]->tag == Constant_TAG) continue;
                decls[new_decls_count++] = process(ctx, old_decls.nodes[i]);
            }
            return root(arena, (Root) { .declarations = nodes(arena, new_decls_count, decls) });
        }
        case RefDecl_TAG: {
            const Node* decl = process(ctx, node->payload.ref_decl.decl);
            if (decl->tag == Constant_TAG) {
                return decl->payload.constant.value;
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* eliminate_constants(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) process)
    };
    const Node* rewritten = process(&ctx, src_program);
    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
