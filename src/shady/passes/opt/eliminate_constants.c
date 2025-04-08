#include "shady/pass.h"
#include "shady/ir/annotation.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

typedef struct {
    Rewriter rewriter;
    bool all;
} Context;

static OpRewriteResult* process(Context* ctx, NodeClass use, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;

    switch (node->tag) {
        case Constant_TAG: {
            Constant payload = node->payload.constant;
            if (!payload.value)
                break;
            if (!ctx->all)
                break;
            // rewrite the constant as the old value if it's used as such
            OpRewriteResult* result = shd_new_rewrite_result(r, NULL);
            shd_rewrite_result_add_mask_rule(result, NcValue, shd_rewrite_op(r, NcValue, "value", payload.value));
            return result;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

static Module* eliminate_constants_(SHADY_UNUSED const CompilerConfig* config, Module* src, bool all) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
        .all = all,
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

Module* shd_pass_eliminate_constants(const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    return eliminate_constants_(config, src, true);
}

Module* shd_pass_eliminate_inlineable_constants(const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    return eliminate_constants_(config, src, false);
}
