#include "pipeline/pipeline_private.h"
#include "shady/ir/debug.h"

#include "portability.h"

#include <string.h>

typedef struct {
    const CompilerConfig* config;
    ExecutionModel em;
} PassConfig;

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Constant_TAG: {
            Node* ncnst = (Node*) shd_recreate_node(&ctx->rewriter, node);
            if (strcmp(shd_get_node_name_safe(ncnst), "SUBGROUP_SIZE") == 0) {
                ncnst->payload.constant.value = shd_uint32_literal(a, ctx->config->target.subgroup_size);
            }
            return ncnst;
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static void specialize_arena_config(ExecutionModel em, TargetConfig* target) {
    target->execution_model = em;
    switch (em) {
        case EmVertex:
        case EmFragment: {
            target->address_spaces[AsShared].allowed = false;
            target->address_spaces[AsSubgroup].allowed = false;
        }
        default: break;
    }
}

static Module* specialize_execution_model(PassConfig* cfg, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    specialize_arena_config(cfg->em, &aconfig.target);

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = cfg->config,
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

static void specialize_execution_model_f(ExecutionModel* em, const CompilerConfig* config, Module** pmod) {
    PassConfig cfg = { .config = config, .em = *em };
    RUN_PASS(specialize_execution_model, &cfg);
}

void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ExecutionModel em) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_execution_model_f, &em, sizeof(ExecutionModel));
}