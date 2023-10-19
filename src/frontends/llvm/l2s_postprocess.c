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
        case Variable_TAG: return var(a, node->payload.var.type ? qualified_type_helper(rewrite_node(&ctx->rewriter, node->payload.var.type), false) : NULL, node->payload.var.name);
        case Function_TAG: {
            Node* decl = (Node*) recreate_node_identity(&ctx->rewriter, node);
            Nodes annotations = decl->payload.fun.annotations;
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            while (an) {
                annotations = append_nodes(a, annotations, an->payload);
                an = an->next;
            }
            decl->payload.fun.annotations = annotations;
            return decl;
        }
        case GlobalVariable_TAG: {
            Node* decl = (Node*) recreate_node_identity(&ctx->rewriter, node);
            Nodes annotations = decl->payload.fun.annotations;
            ParsedAnnotation* an = find_annotation(ctx->p, node);
            while (an) {
                annotations = append_nodes(a, annotations, an->payload);
                an = an->next;
            }
            return decl;
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
