#include "shady/pass.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
} Context;

static OpRewriteResult process(Context* ctx, NodeClass use, String name, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            if (node->payload.global_variable.address_space == AsGeneric) {
                AddressSpace dst_as = AsGlobal;
                const Type* t = shd_rewrite_op(&ctx->rewriter, NcType, "type", node->payload.global_variable.type);
                if (use == NcValue) {
                    const Type* dst_t = ptr_type(a, (PtrType) { .pointed_type = t, .address_space = AsGeneric });
                    const Node* new_global = shd_rewrite_op(r, NcDeclaration, "decl", node);
                    const Node* converted = prim_op_helper(a, convert_op, shd_singleton(dst_t), shd_singleton(new_global));
                    return (OpRewriteResult) { converted, NcValue };
                } else {
                    Node* new_global = global_variable_helper(r->dst_module, shd_rewrite_ops(r, NcAnnotation, "annotations", node->payload.global_variable.annotations), t, node->payload.global_variable.name, dst_as, payload.is_ref);
                    shd_register_processed_mask(r, node, new_global, ~NcValue);
                    shd_recreate_node_body(r, node, new_global);
                    return (OpRewriteResult) { new_global, ~NcValue };
                }
            }
            break;
        }
        default: break;
    }

    return (OpRewriteResult) { shd_recreate_node(&ctx->rewriter, node), 0 };
}

Module* shd_pass_lower_generic_globals(SHADY_UNUSED const CompilerConfig* config, Module* src) {
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
