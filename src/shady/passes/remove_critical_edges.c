#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
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
            set_abstraction_body(new_block, jump_helper(a, rewrite_node(r, payload.target), rewrite_nodes(r, payload.args), get_abstraction_mem(new_block)));
            return jump_helper(a, new_block, shd_empty(a), rewrite_node(r, payload.mem));
        }
        default: break;
    }

    return recreate_node_identity(r, node);
}

Module* remove_critical_edges(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
