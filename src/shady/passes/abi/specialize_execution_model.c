#include "shady/pass.h"
#include "shady/pipeline/pipeline.h"
#include "shady/ir/debug.h"

#include "portability.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

static void specialize_arena_config(ExecutionModel em, TargetConfig* target) {
    target->execution_model = em;
    switch (em) {
        case EmVertex:
        case EmFragment: {
            target->memory.address_spaces[AsShared].allowed = false;
            target->memory.address_spaces[AsSubgroup].allowed = false;
        }
        default: break;
    }
}

static Module* specialize_execution_model(SHADY_UNUSED const CompilerConfig* config, ExecutionModel* em, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    specialize_arena_config(*em, &aconfig.target);

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

static void specialize_execution_model_f(ExecutionModel* em, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(specialize_execution_model, em);
}

void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ExecutionModel em) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_execution_model_f, &em, sizeof(ExecutionModel));
}