#include "passes.h"

#include "portability.h"
#include "log.h"

#include "../ir_private.h"
#include "../rewrite.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    if (!node) return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

static void specialize_arena_config(const CompilerConfig* config, Module* m, ArenaConfig* target) {
    switch (config->specialization.execution_model) {
        case EmVertex:
        case EmFragment: {
            target->address_spaces[AsShared].allowed = false;
            target->address_spaces[AsSubgroup].allowed = false;
        }
        default: break;
    }
}

Module* specialize_execution_model(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    specialize_arena_config(config, src, &aconfig);
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
