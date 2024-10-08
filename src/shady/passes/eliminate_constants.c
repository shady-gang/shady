#include "shady/pass.h"
#include "shady/ir/annotation.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

typedef struct {
    Rewriter rewriter;
    bool all;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Constant_TAG:
            if (!node->payload.constant.value)
                break;
            if (!ctx->all && !shd_lookup_annotation(node, "Inline"))
                break;
            return NULL;
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            if (decl->tag == Constant_TAG && decl->payload.constant.value) {
                return shd_rewrite_node(&ctx->rewriter, decl->payload.constant.value);
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

static Module* eliminate_constants_(SHADY_UNUSED const CompilerConfig* config, Module* src, bool all) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .all = all,
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

Module* shd_pass_eliminate_constants(const CompilerConfig* config, Module* src) {
    return eliminate_constants_(config, src, true);
}

Module* shd_pass_eliminate_inlineable_constants(const CompilerConfig* config, Module* src) {
    return eliminate_constants_(config, src, false);
}
