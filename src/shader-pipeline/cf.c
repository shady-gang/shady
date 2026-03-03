#include "shader_pipeline.h"

#include "shady/passes/fncall_passes.h"
#include "shady/passes/stack_passes.h"
#include "shady/passes/opt_passes.h"
#include "shady/passes/scf_passes.h"

#include "portability.h"
#include "log.h"

void shd_add_scheduler_source(const CompilerConfig* config, Module* dst);

static CompilationResult remove_indirect_calls(const TargetConfig* target_config, const CompilerConfig* config, Module** pmod) {
    if (!target_config->capabilities.native_stack)
        SHADY_APPLY_REWRITE_PASS(shd_pass_setup_stack_frames)
    if (!config->hacks.force_join_point_lifting)
        SHADY_APPLY_REWRITE_PASS(shd_pass_mark_leaf_functions)

    if (!target_config->capabilities.native_fncalls) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_callf)
        SHADY_APPLY_REWRITE_PASS(shd_pass_inline)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lift_indirect_targets)

        if (config->dynamic_scheduling) {
            shd_add_scheduler_source(config, *pmod);
        }

        // run this again so the scheduler source is left alone
        SHADY_APPLY_REWRITE_PASS(shd_pass_mark_leaf_functions)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_dynamic_control)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_tailcalls)
    }

    return CompilationNoError;
}

void shd_pipeline_add_fncall_emulation(ShdPipeline pipeline, const TargetConfig* target_config) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) remove_indirect_calls, target_config, sizeof(TargetConfig));
}

static CompilationResult restructure(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    SHADY_APPLY_REWRITE_PASS(shd_pass_restructurize)

    return CompilationNoError;
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, restructure, NULL, 0);
}
