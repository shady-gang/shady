#include "shady/ir.h"

#include "../rewrite.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
} Context;

#include "dict.h"

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
        case Constant_TAG: {
            //const Node* value = process(ctx, node->payload.constant.value);
            //register_processed(&ctx->rewriter, node, value);
            //return value;
        }
        case GlobalVariable_TAG:
        case Function_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
            recreate_decl_body_identity(&ctx->rewriter, node, new);
            return new;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* eliminate_constants(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    Context ctx = {
        .rewriter = {
            .src_arena = src_arena,
            .dst_arena = dst_arena,
            .rewrite_fn = (RewriteFn) process,
            .processed = done,
        },
    };

    const Node* rewritten = process(&ctx, src_program);

    destroy_dict(done);
    return rewritten;
}
