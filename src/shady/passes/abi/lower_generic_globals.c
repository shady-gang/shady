#include "shady/pass.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static OpRewriteResult* process(Context* ctx, NodeClass use, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space == AsGeneric) {
                if (shd_lookup_annotation(node, "Exported") || shd_lookup_annotation(node, "IO") || shd_lookup_annotation(node, "Builtin"))
                    break;
                const Type* t = shd_rewrite_op(&ctx->rewriter, NcType, "type", node->payload.global_variable.type);
                GlobalVariable payload = node->payload.global_variable;
                // we need to re-promote this in case it was demoted, because we will cast the global to a generic pointer still
                // if this can be optimized away later we win
                payload.is_ref = false;
                payload = shd_rewrite_global_head_payload(r, payload);
                payload.address_space = AsPrivate;
                Node* new_global = shd_global_var(r->dst_module, payload);
                const Node* converted = generic_ptr_cast_helper(a, new_global);
                shd_register_processed(r, node, converted);
                shd_recreate_node_body(r, node, new_global);
                shd_rewrite_annotations(r, node, new_global);
                return shd_new_rewrite_result(r, converted);
            }
            break;
        }
        default: break;
    }

    return shd_new_rewrite_result(r, shd_recreate_node(r, node));
}

Module* shd_pass_lower_generic_globals(SHADY_UNUSED const CompilerConfig* config, SHADY_UNUSED const void* unused, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_op_rewriter(src, dst, (RewriteOpFn) process),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
