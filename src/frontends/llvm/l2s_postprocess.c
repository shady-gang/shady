#include "l2s_private.h"

#include "portability.h"
#include "../shady/rewrite.h"
#include "../shady/type.h"

typedef struct {
    Rewriter rewriter;
    Parser* p;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Variable_TAG: return var(a, qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.var.type), false), node->payload.var.name);
        case Function_TAG: {
            Node* fun = recreate_node_identity(&ctx->rewriter, node);
            ParsedAnnotationContents* ep_type = find_annotation(ctx->p, node, EntryPointAnnot);
            if (ep_type) {
                fun->payload.fun.annotations = append_nodes(a, fun->payload.fun.annotations, annotation_value(a, (AnnotationValue) {
                    .name = "EntryPoint",
                    .value = string_lit_helper(a, ep_type->payload.entry_point_type)
                }));
            }
            return fun;
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

void postprocess(Parser* p, Module* src, Module* dst) {
    assert(src != dst);
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .p = p,
    };

    ctx.rewriter.config.process_variables = true;
    // ctx.rewriter.config.search_map = false;
    // ctx.rewriter.config.write_map = false;

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
