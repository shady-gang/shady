#include "shader_pipeline.h"

#include "shady/passes/fncall_passes.h"
#include "shady/passes/stack_passes.h"
#include "shady/passes/opt_passes.h"
#include "shady/passes/scf_passes.h"

#include "portability.h"
#include "log.h"

void shd_add_scheduler_source(const CompilerConfig* config, const TargetConfig* target, const ShaderLoweringConfig*, Module* dst);

typedef struct {
    const TargetConfig* target_config;
    const ShaderLoweringConfig* lowering_config;
} S;

static ShdResult remove_indirect_calls(const S* s, const CompilerConfig* config, Module** pmod) {
    const TargetConfig* target = s->target_config;
    if (!target->capabilities.native_stack)
        SHADY_APPLY_REWRITE_PASS(shd_pass_setup_stack_frames)
    if (!config->hacks.force_join_point_lifting)
        SHADY_APPLY_REWRITE_PASS(shd_pass_mark_leaf_functions)

    //if (!target->capabilities.native_fncalls) {
    if (s->lowering_config->function_call_lowering == FCL_SoftwareScheduler) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_callf)
        SHADY_APPLY_REWRITE_PASS(shd_pass_inline)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lift_indirect_targets)

        if (s->lowering_config->exec_model_info) {
            shd_add_scheduler_source(config, target, s->lowering_config, *pmod);
        } else {
            shd_log_fmt(ERROR, "Using the software scheduler requires an entry point to be known.\n");
            shd_log_fmt(ERROR, "Provided a source file with a single entry point or use --entry-point to name the one you wish to specialize on.\n");
            shd_error_die();
        }

        // run this again so the scheduler source is left alone
        SHADY_APPLY_REWRITE_PASS(shd_pass_mark_leaf_functions)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_dynamic_control)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_tailcalls)
    }

    return SHD_SUCCESS;
}

void shd_pipeline_add_fncall_emulation(ShdPipeline pipeline, const ShaderLoweringConfig* lowering_config, const TargetConfig* target_config) {
    S s = {
        .lowering_config = lowering_config,
        .target_config = target_config,
    };
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) remove_indirect_calls, &s, sizeof(S));
}

static ShdResult restructure(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    SHADY_APPLY_REWRITE_PASS(shd_pass_restructurize)

    return SHD_SUCCESS;
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, restructure, NULL, 0);
}
