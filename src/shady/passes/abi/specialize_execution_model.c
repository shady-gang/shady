#include "shady/pass.h"
#include "shady/pipeline/pipeline.h"
#include "shady/ir/debug.h"

#include "portability.h"

#include <string.h>

typedef struct {
    Rewriter rewriter;
} Context;

static Module* specialize_execution_model(SHADY_UNUSED const CompilerConfig* config, ShdExecutionModel* em, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.target.execution_model = *em;
    shd_target_apply_execution_model_restrictions(&aconfig.target);

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) shd_recreate_node),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}

static CompilationResult specialize_execution_model_f(ShdExecutionModel* em, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(specialize_execution_model, em);
    return CompilationNoError;
}

void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ShdExecutionModel em) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_execution_model_f, &em, sizeof(ShdExecutionModel));
}