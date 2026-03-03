#include "shader_pipeline.h"

#include "shady/passes/mem_passes.h"
#include "shady/passes/ptr_passes.h"
#include "shady/passes/stack_passes.h"
#include "shady/passes/io_passes.h"
#include "shady/passes/group_passes.h"

static void lower_memory(const TargetConfig* target, const CompilerConfig* config, Module** pmod) {
    SHADY_APPLY_REWRITE_PASS(shd_pass_promote_io_variables)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_logical_pointers)

    if (!target->capabilities.native_memcpy) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_memcpy)
    }

    if (!target->capabilities.native_stack) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_alloca)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_stack_access)
    }
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_lea, &target->memory)
    if (!target->memory.address_spaces[AsGeneric].allowed) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_generic_ptrs)
    }
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_physical_memory, &target->memory)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_subgroup_vars)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_memory_layout)
    if (config->lower.decay_ptrs)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_decay_ptrs)
}

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, const TargetConfig* tgt) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) lower_memory, (void*) tgt, sizeof(TargetConfig));
}
