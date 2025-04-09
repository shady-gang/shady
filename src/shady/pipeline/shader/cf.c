#include "pipeline/pipeline_private.h"

#include "passes/passes.h"
#include "portability.h"
#include "log.h"

/// Implements stack frames: saves the stack size on function entry and restores it upon exit
RewritePass shd_pass_setup_stack_frames;

/// Tags all functions that don't need special handling
RewritePass shd_pass_mark_leaf_functions;

/// Lowers calls to stack saves and forks, lowers returns to stack pops and joins
RewritePass shd_pass_lower_callf;

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass shd_pass_lift_indirect_targets;

/// Wires up intrinsics to the built-in scheduler code
RewritePass shd_pass_lower_dynamic_control;
/// Creates a top-level function
RewritePass shd_pass_lower_tailcalls;

void shd_add_scheduler_source(const CompilerConfig* config, Module* dst);

static CompilationResult remove_indirect_calls(const TargetConfig* target_config, const CompilerConfig* config, Module** pmod) {
    if (!target_config->capabilities.native_stack)
        RUN_PASS(shd_pass_setup_stack_frames, config)
    if (!config->hacks.force_join_point_lifting)
        RUN_PASS(shd_pass_mark_leaf_functions, config)

    if (!target_config->capabilities.native_fncalls) {
        RUN_PASS(shd_pass_lower_callf, config)
        RUN_PASS(shd_pass_inline, config)
        RUN_PASS(shd_pass_lift_indirect_targets, config)

        if (config->dynamic_scheduling) {
            shd_add_scheduler_source(config, *pmod);
        }

        // run this again so the scheduler source is left alone
        RUN_PASS(shd_pass_mark_leaf_functions, config)
        RUN_PASS(shd_pass_lower_dynamic_control, config)
        RUN_PASS(shd_pass_lower_tailcalls, config)
    }

    return CompilationNoError;
}

void shd_pipeline_add_fncall_emulation(ShdPipeline pipeline, TargetConfig target_config) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) remove_indirect_calls, &target_config, sizeof(TargetConfig));
}

static CompilationResult restructure(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_restructurize, config)

    return CompilationNoError;
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, restructure, NULL, 0);
}

static CompilationResult normalize_input_cf(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    if (config->input_cf.has_scope_annotations) {
        // RUN_PASS(shd_pass_scope_heuristic)
        // RUN_PASS(shd_pass_lift_everything, config)
        RUN_PASS(shd_pass_lcssa, config)
        RUN_PASS(shd_pass_scope2control, config)
    } else if (config->input_cf.restructure_with_heuristics) {
        RUN_PASS(shd_pass_remove_critical_edges, config)
        RUN_PASS(shd_pass_lcssa, config)
        // RUN_PASS(shd_pass_lift_everything)
        RUN_PASS(shd_pass_reconvergence_heuristics, config)
    }

    return CompilationNoError;
}

void shd_pipeline_add_normalize_input_cf(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, normalize_input_cf, NULL, 0);
}
