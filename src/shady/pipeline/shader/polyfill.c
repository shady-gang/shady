#include "pipeline/pipeline_private.h"

#include "passes/passes.h"
#include "portability.h"

/// Emulates unsupported subgroup operations using subgroup memory
RewritePass shd_pass_lower_subgroup_ops;
/// Emulates unsupported integer datatypes and operations
RewritePass shd_pass_lower_int;
RewritePass shd_pass_lower_fill;
RewritePass shd_pass_lower_nullptr;

static void polyfills(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_lower_int, config)

    RUN_PASS(shd_pass_lower_mask, config)
    RUN_PASS(shd_pass_lower_fill, config)
    RUN_PASS(shd_pass_lower_nullptr, config)
    RUN_PASS(shd_pass_lower_subgroup_ops, config)
}

void shd_pipeline_add_polyfills(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) polyfills, NULL, 0);
}
