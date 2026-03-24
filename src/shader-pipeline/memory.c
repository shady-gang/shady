#include "shader_pipeline.h"

#include "shady/passes/mem_passes.h"
#include "shady/passes/ptr_passes.h"
#include "shady/passes/stack_passes.h"
#include "shady/passes/io_passes.h"
#include "shady/passes/group_passes.h"

typedef struct {
    const TargetConfig* target_config;
    const ShaderLoweringConfig* lowering_config;
    uint32_t subgroups_per_wg;
} S;

static void lower_memory(const S* s, const CompilerConfig* config, Module** pmod) {
    const TargetConfig* target = s->target_config;
    ShdExecutionModel em = ShdExecutionModelNone;
    if (s->lowering_config->exec_model_info)
        em = s->lowering_config->exec_model_info->execution_model;

    SHADY_APPLY_REWRITE_PASS(shd_pass_promote_io_variables)
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_logical_pointers)

    if (!target->capabilities.native_memcpy) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_memcpy)
    }

    if (!target->capabilities.native_stack) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_alloca)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_stack_access, s->lowering_config->per_thread_stack_size)
    }
    //SHADY_APPLY_REWRITE_PASS(shd_pass_lower_lea, &target->ptr_model)
    if (!target->ptr_model.address_spaces[AsGeneric].allowed) {
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_generic_ptrs)
    }

    PtrModel ptr_model = target->ptr_model;
    ptr_model.address_spaces[AsCode].physical = true;
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_physical_memory, &ptr_model, em)
    if (s->lowering_config->exec_model_info && shd_is_execution_model_workgroup_based(s->lowering_config->exec_model_info->execution_model)) {
        uint32_t subgroups_per_wg = 1;
        shd_get_num_subgroups_per_workgroups(s->lowering_config->exec_model_info, target->subgroup_size, &subgroups_per_wg);
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_subgroup_vars, s->subgroups_per_wg)
    }
    SHADY_APPLY_REWRITE_PASS(shd_pass_lower_memory_layout)
    if (config->lower.decay_ptrs)
        SHADY_APPLY_REWRITE_PASS(shd_pass_lower_decay_ptrs)
}

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, const ShaderLoweringConfig* lowering_config, const TargetConfig* target_config, uint32_t subgroups_per_wg) {
    S s = {
        .lowering_config = lowering_config,
        .target_config = target_config,
        .subgroups_per_wg = subgroups_per_wg,
    };
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) lower_memory, (void*) &s, sizeof(S));
}
