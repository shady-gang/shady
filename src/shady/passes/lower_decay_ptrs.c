#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PtrType_TAG: {
            const Node* arr_t = node->payload.ptr_type.pointed_type;
            if (arr_t->tag == ArrType_TAG && !arr_t->payload.arr_type.size) {
                return ptr_type(arena, (PtrType) {
                    .pointed_type = shd_rewrite_node(&ctx->rewriter, arr_t->payload.arr_type.element_type),
                    .address_space = node->payload.ptr_type.address_space,
                });
            }
            break;
        }
        default: break;
    }

    rebuild:
    return shd_recreate_node(&ctx->rewriter, node);
}

Module* lower_decay_ptrs(const CompilerConfig* config, Module* src) {
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
