#include "pipeline/pipeline_private.h"

#include "passes/passes.h"
#include "portability.h"
#include "log.h"

static CompilationResult remove_indirect_calls(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_setup_stack_frames, config)
    if (!config->hacks.force_join_point_lifting)
        RUN_PASS(shd_pass_mark_leaf_functions, config)

    RUN_PASS(shd_pass_lower_callf, config)
    RUN_PASS(shd_pass_inline, config)
    RUN_PASS(shd_pass_lift_indirect_targets, config)

    return CompilationNoError;
}

void shd_pipeline_add_remove_indirect_calls(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, remove_indirect_calls, NULL, 0);
}

#include "../frontend/slim/parser.h"
#include "shady_scheduler_src.h"

static void add_scheduler_source(const CompilerConfig* config, Module* dst) {
    SlimParserConfig pconfig = {
            .front_end = true,
    };
    Module* builtin_scheduler_mod = shd_parse_slim_module(config, &pconfig, shady_scheduler_src, "builtin_scheduler");
    shd_debug_print("Adding builtin scheduler code");
    shd_module_link(dst, builtin_scheduler_mod);
    shd_destroy_ir_arena(shd_module_get_arena(builtin_scheduler_mod));
}

static CompilationResult emulate_tailcalls(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    if (config->dynamic_scheduling) {
        add_scheduler_source(config, *pmod);
    }

    // run this again so the scheduler source is left alone
    RUN_PASS(shd_pass_mark_leaf_functions, config)
    RUN_PASS(shd_pass_lower_tailcalls, config)
    RUN_PASS(shd_pass_lower_mask, config)

    return CompilationNoError;
}

void shd_pipeline_add_tailcall_elimination(ShdPipeline pipeline) {
    shd_pipeline_add_step(pipeline, emulate_tailcalls, NULL, 0);
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
        RUN_PASS(shd_pass_lift_everything, config)
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
