#include "shady/pass.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
} Context;

static OpRewriteResult process_op(Context* ctx, NodeClass op_class, SHADY_UNUSED String op_name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        // All decls map to refdecl/fnaddr
        case Constant_TAG:
        case GlobalVariable_TAG: {
            if (op_class == NcValue)
                return (OpRewriteResult) { ref_decl_helper(a, shd_rewrite_op(r, NcDeclaration, "decl", node)), NcValue };
            break;
        }
        case Function_TAG: {
            if (op_class == NcValue)
                return (OpRewriteResult) { fn_addr_helper(a, shd_rewrite_op(r, NcDeclaration, "decl", node)), NcValue };
            break;
        }
        default:
            break;
    }

    return (OpRewriteResult) { shd_recreate_node(r, node), 0 };
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
