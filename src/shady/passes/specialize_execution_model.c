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
    CompilerConfig* config;
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

void specialize_configurations_for_execution_model(Module* m, ArenaConfig* target, CompilerConfig* config) {
    switch (config->specialization.execution_model) {
        case EmFragment: {
            target->allow_subgroup_memory = false;
        }
        default: break;
    }
}

void specialize_execution_model(CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .config = config,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
