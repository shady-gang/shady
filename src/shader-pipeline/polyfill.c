#include "shader_pipeline.h"

#include "shady/passes/polyfill_passes.h"
#include "shady/passes/ptr_passes.h"
#include "shady/passes/group_passes.h"

#include "portability.h"

static void polyfills(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_int)

    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_mask)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_fill)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_nullptr)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_subgroup_ops)
}

void shd_pipeline_add_polyfills(ShdPipeline pipeline, const TargetConfig* tgt) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) polyfills, NULL, 0);
}
