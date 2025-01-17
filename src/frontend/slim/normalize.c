#include "shady/pass.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
} Context;

static OpRewriteResult* process_op(Context* ctx, NodeClass op_class, SHADY_UNUSED String op_name, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case Function_TAG: {
            Node* new = shd_recreate_node_head_(r, old);
            OpRewriteResult* result = shd_new_rewrite_result(r, new);
            shd_rewrite_result_add_mask_rule(result, NcValue, fn_addr_helper(a, new));
            shd_register_processed_result(r, old, result);
            shd_recreate_node_body(r, old, new);
            shd_rewrite_annotations(r, old, new);
            return result;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, old));
}

Module* slim_pass_normalize(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.check_op_classes = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process_op),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
