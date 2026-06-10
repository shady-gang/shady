#include "shader_pipeline.h"
#include "shady/pass.h"

#include "portability.h"
#include "log.h"

static void shd_pipeline_add_feature_lowering(ShdPipeline pipeline, const ShaderLoweringConfig* lowering_config, const TargetConfig* target_config, uint32_t subgroups_per_wg) {
    shd_pipeline_add_memory_lowering(pipeline, lowering_config, target_config, subgroups_per_wg);
    shd_pipeline_add_polyfills(pipeline, lowering_config);
    // TODO: move that one to the backends.
    shd_pipeline_add_restructure_cf(pipeline);
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);

/// questionably useful pass that updates the exec mask size
static Module* specialize_target_config(SHADY_UNUSED const CompilerConfig* config, Module* src, TargetConfig* target_config) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    MachineRules new_rules = get_machine_rules_from_target_config(target_config);
    aconfig.rules.exec_mask_size = new_rules.exec_mask_size;

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Rewriter r = shd_create_importer(src, dst);
    shd_rewrite_module(&r);
    shd_destroy_rewriter(&r);
    return dst;
}

static ShdResult specialize_target_config_step(TargetConfig* target_config, const CompilerConfig* config, Module** pmod) {
    SHADY_APPLY_REWRITE_PASS(specialize_target_config, (void*) target_config);
    return SHD_SUCCESS;
}

void shd_pipeline_add_target_specialization(ShdPipeline pipeline, const TargetConfig* target_config) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_target_config_step, (void*) target_config, sizeof(TargetConfig));
}

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, const ShaderLoweringConfig* lowering_config, const TargetConfig* target) {
    shd_pipeline_add_target_specialization(pipeline, target);

    uint32_t subgroups_per_wg = 1;
    if (lowering_config->exec_model_info && shd_is_execution_model_workgroup_based(lowering_config->exec_model_info->execution_model)) {
        bool ok = shd_get_num_subgroups_per_workgroups(lowering_config->exec_model_info, target->subgroup_size, &subgroups_per_wg);
        if (!ok)
            shd_warn_print("Could not determine number of subgroups per workgroup, defaulting to one.\n");
    }

    shd_pipeline_add_fncall_emulation(pipeline, lowering_config, target, subgroups_per_wg);
    shd_pipeline_add_feature_lowering(pipeline, lowering_config, target, subgroups_per_wg);
}