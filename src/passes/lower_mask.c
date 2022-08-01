#include "shady/ir.h"

#include "dict.h"

#include "../rewrite.h"
#include "../portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    if (node->tag == MaskType_TAG)
        return int64_type(ctx->rewriter.dst_arena);
    else if (is_declaration(node->tag)) {
        Node* new = recreate_decl_header_identity(&ctx->rewriter, node);
        recreate_decl_body_identity(&ctx->rewriter, node, new);
        return new;
    } else return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

const Node* lower_mask(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);

    Context context = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .processed = done,
            .rewrite_fn = (RewriteFn) process,
        }
    };

    const Node* new = rewrite_node(&context.rewriter, src_program);

    destroy_dict(done);

    return new;
}