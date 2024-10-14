#include "shady/pass.h"
#include "shady/visit.h"
#include "shady/ir/stack.h"

#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;

    const CompilerConfig* config;
    const Node* stack_size_on_entry;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    switch (node->tag) {
        case Function_TAG: {
            Node* fun = shd_recreate_node_head(r, node);
            Context ctx2 = *ctx;
            ctx2.disable_lowering = shd_lookup_annotation_with_string_payload(node, "DisablePass", "setup_stack_frames") || ctx->config->per_thread_stack_size == 0;

            BodyBuilder* bb = shd_bld_begin(a, shd_get_abstraction_mem(fun));
            if (!ctx2.disable_lowering) {
                ctx2.stack_size_on_entry = shd_bld_get_stack_size(bb);
                shd_set_value_name((Node*) ctx2.stack_size_on_entry, shd_format_string_arena(a->arena, "saved_stack_ptr_entering_%s", shd_get_abstraction_name(fun)));
            }
            shd_register_processed(&ctx2.rewriter, shd_get_abstraction_mem(node), shd_bb_mem(bb));
            if (node->payload.fun.body)
                shd_set_abstraction_body(fun, shd_bld_finish(bb, shd_rewrite_node(&ctx2.rewriter, node->payload.fun.body)));
            else
                shd_bld_cancel(bb);
            return fun;
        }
        case Return_TAG: {
            Return payload = node->payload.fn_ret;
            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            if (!ctx->disable_lowering) {
                assert(ctx->stack_size_on_entry);
                // Restore SP before calling exit
                shd_bld_set_stack_size(bb, ctx->stack_size_on_entry);
            }
            return shd_bld_finish(bb, fn_ret(a, (Return) {
                .mem = shd_bb_mem(bb),
                .args = shd_rewrite_nodes(r, payload.args),
            }));
        }
        default: break;
    }
    return shd_recreate_node(r, node);
}

Module* shd_pass_setup_stack_frames(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
