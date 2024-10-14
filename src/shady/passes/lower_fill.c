#include "shady/pass.h"

#include "../ir_private.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Fill_TAG: {
            const Type* composite_t = shd_rewrite_node(r, node->payload.fill.type);
            size_t actual_size = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_fill_type_size(composite_t)), false);
            const Node* value = shd_rewrite_node(r, node->payload.fill.value);
            LARRAY(const Node*, copies, actual_size);
            for (size_t i = 0; i < actual_size; i++) {
                copies[i] = value;
            }
            return composite_helper(a, composite_t, shd_nodes(a, actual_size, copies));
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_pass_lower_fill(SHADY_UNUSED const CompilerConfig* config, Module* src) {
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
