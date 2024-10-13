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
        case RefDecl_TAG: {
            // make sure we rewrite the decl first, and then look if it rewrote the ref to it!
            shd_rewrite_node(r, node->payload.ref_decl.decl);
            const Node** f = shd_search_processed(r, node);
            if (f) return *f;
            break;
        }
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space == AsGeneric) {
                AddressSpace dst_as = AsGlobal;
                const Type* t = shd_rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                Node* new_global = global_var(ctx->rewriter.dst_module, shd_rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), t, node->payload.global_variable.name, dst_as);
                shd_register_processed(&ctx->rewriter, node, new_global);

                const Type* dst_t = ptr_type(a, (PtrType) { .pointed_type = t, .address_space = AsGeneric });
                const Node* converted = prim_op_helper(a, convert_op, shd_singleton(dst_t), shd_singleton(ref_decl_helper(a, new_global)));
                shd_register_processed(&ctx->rewriter, ref_decl_helper(node->arena, node), converted);
                return new_global;
            }
            break;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lower_generic_globals(SHADY_UNUSED const CompilerConfig* config, Module* src) {
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
