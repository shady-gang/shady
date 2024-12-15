#include "pipeline/pipeline_private.h"

#include "passes/passes.h"
#include "portability.h"

static void lower_memory(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_promote_io_variables, config)
    RUN_PASS(shd_pass_lower_logical_pointers, config)

    if (config->lower.emulate_physical_memory) {
        RUN_PASS(shd_pass_lower_alloca, config)
    }
    RUN_PASS(shd_pass_lower_stack, config)
    RUN_PASS(shd_pass_lower_memcpy, config)
    RUN_PASS(shd_pass_lower_lea, config)
    RUN_PASS(shd_pass_lower_generic_globals, config)
    if (config->lower.emulate_generic_ptrs) {
        RUN_PASS(shd_pass_lower_generic_ptrs, config)
    }
    if (config->lower.emulate_physical_memory) {
        RUN_PASS(shd_pass_lower_physical_ptrs, config)
    }
    RUN_PASS(shd_pass_lower_subgroup_vars, config)
    RUN_PASS(shd_pass_lower_memory_layout, config)
    if (config->lower.decay_ptrs)
        RUN_PASS(shd_pass_lower_decay_ptrs, config)
}

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) lower_memory, NULL, 0);
}
