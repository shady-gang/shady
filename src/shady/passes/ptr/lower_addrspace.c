#include "shady/passes/ptr_passes.h"

#include "portability.h"
#include "shady/ir/decl.h"

typedef struct {
    Rewriter rewriter;
    AddressSpace from;
    AddressSpace to;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PtrType_TAG: {
            PtrType payload = node->payload.ptr_type;
            if (payload.address_space == ctx->from) {
                return ptr_type(a, (PtrType) {
                    .pointed_type = shd_rewrite_node(r, payload.pointed_type),
                    .address_space = ctx->to,
                });
            }
            break;
        }
        case LocalAlloc_TAG: {
            // TODO: only if LocalAlloc allocates in Function!
            assert(ctx->from == AsFunction);
            const Node* n = shd_recreate_node(r, node);
            n = mem_and_value_helper(a, n, addr_space_cast_helper(a, n, ctx->to));
            return n;
        }
        case GlobalVariable_TAG: {
            GlobalVariable payload = node->payload.global_variable;
            if (payload.address_space != ctx->from)
                break;
            payload.address_space = ctx->to;
            payload = shd_rewrite_global_head_payload(r, payload);
            Node* n = shd_global_var(r->dst_module, payload);
            shd_register_processed(r, node, n);
            shd_recreate_node_body(r, node, n);
            return n;
        }
        default: break;
    }

    return shd_recreate_node(r, node);
}

Module* shd_pass_lower_addrspace(SHADY_UNUSED const CompilerConfig* config, Module* src, AddressSpace from, AddressSpace to) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .from = from,
        .to = to,
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
