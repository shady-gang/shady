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
        case RefDecl_TAG: {
            // make sure we rewrite the decl first, and then look if it rewrote the ref to it!
            rewrite_node(r, node->payload.ref_decl.decl);
            const Node** f = search_processed(r, node);
            if (f) return *f;
            break;
        }
        case GlobalVariable_TAG: {
            if (node->payload.global_variable.address_space == AsGeneric) {
                AddressSpace dst_as = AsGlobal;
                const Type* t = rewrite_node(&ctx->rewriter, node->payload.global_variable.type);
                Node* new_global = global_var(ctx->rewriter.dst_module, rewrite_nodes(&ctx->rewriter, node->payload.global_variable.annotations), t, node->payload.global_variable.name, dst_as);
                register_processed(&ctx->rewriter, node, new_global);

                const Type* dst_t = ptr_type(a, (PtrType) { .pointed_type = t, .address_space = AsGeneric });
                const Node* converted = prim_op_helper(a, convert_op, singleton(dst_t), singleton(ref_decl_helper(a, new_global)));
                register_processed(&ctx->rewriter, ref_decl_helper(node->arena, node), converted);
                return new_global;
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_generic_globals(SHADY_UNUSED const CompilerConfig* config, Module* src) {
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
