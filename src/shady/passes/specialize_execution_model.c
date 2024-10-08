#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"

#include "portability.h"
#include "log.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Constant_TAG: {
            Node* ncnst = (Node*) shd_recreate_node(&ctx->rewriter, node);
            if (strcmp(get_declaration_name(ncnst), "SUBGROUP_SIZE") == 0) {
                ncnst->payload.constant.value = shd_uint32_literal(a, ctx->config->specialization.subgroup_size);
            }
            return ncnst;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
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

Module* shd_pass_specialize_execution_model(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    specialize_arena_config(config, src, &aconfig);
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    size_t subgroup_size = config->specialization.subgroup_size;
    assert(subgroup_size);

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
