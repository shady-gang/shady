#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../ir_private.h"
#include "../type.h"
#include "../rewrite.h"
#include "../transform/ir_gen_helpers.h"

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case PtrType_TAG: {
            const Node* arr_t = node->payload.ptr_type.pointed_type;
            if (arr_t->tag == ArrType_TAG && !arr_t->payload.arr_type.size) {
                return ptr_type(arena, (PtrType) {
                    .pointed_type = rewrite_node(&ctx->rewriter, arr_t->payload.arr_type.element_type),
                    .address_space = node->payload.ptr_type.address_space,
                });
            }
            break;
        }
        default: break;
    }

    rebuild:
    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lower_decay_ptrs(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
