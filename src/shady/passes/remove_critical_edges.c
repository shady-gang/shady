#include "shady/pass.h"

#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Jump_TAG: {
            Jump payload = node->payload.jump;
            Node* new_block = basic_block(a, shd_empty(a), NULL);
            shd_set_abstraction_body(new_block, jump_helper(a, shd_get_abstraction_mem(new_block),
                                                            shd_rewrite_node(r, payload.target),
                                                            shd_rewrite_nodes(r, payload.args)));
            return jump_helper(a, shd_rewrite_node(r, payload.mem), new_block, shd_empty(a));
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_pass_remove_critical_edges(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
