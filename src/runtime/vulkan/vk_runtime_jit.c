#include "shady/jit/vulkan.h"

#include "shady/pipeline/pipeline.h"
#include "shady/pass.h"
#include "shady/pipeline/shader_pipeline.h"

#include "log.h"

void shd_jit_vk_get_compiler_config_for_device(const ShadyVkrPhysicalDeviceCaps* caps, const TargetConfig* target_config, SPVBackendConfig* spv_config, CompilerConfig* config) {
    assert(caps->subgroup_size.max > 0);
    // config.per_thread_stack_size = ...

    *spv_config = shd_default_spirv_backend_config();
    shd_spv_apply_target_config(spv_config, target_config);

    spv_config->target_version.major = caps->spirv_version.major;
    spv_config->target_version.minor = caps->spirv_version.minor;

    if (!caps->features.subgroup_extended_types.shaderSubgroupExtendedTypes)
        config->lower.emulate_subgroup_ops_extended_types = true;

    config->lower.int64 = !caps->features.base.features.shaderInt64;

    if (caps->implementation.is_moltenvk) {
        shd_warn_print("Hack: MoltenVK says they supported subgroup extended types, but it's a lie. 64-bit types are unaccounted for !\n");
        config->lower.emulate_subgroup_ops_extended_types = true;
        shd_warn_print("Hack: MoltenVK does not support pointers to unsized arrays properly.\n");
        config->lower.decay_ptrs = true;
        spv_config->hacks.avoid_spirv_cross_broken_bda_pointers = true;
    }
    if (caps->properties.driver_properties.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        shd_warn_print("Hack: NVidia somehow has unreliable broadcast_first. Emulating it with shuffles seemingly fixes the issue.\n");
        spv_config->hacks.shuffle_instead_of_broadcast_first = true;
    }
}

CompilationResult shd_jit_vk_compile_module(Module** module, const TargetConfig* target_config, const SPVBackendConfig* backend_config, const CompilerConfig* compiler_config) {
    *module = shd_import(compiler_config, *module);

    ShdPipeline pipeline = shd_create_empty_pipeline();
    shd_pipeline_add_shader_target_lowering(pipeline, target_config, compiler_config);
    shd_pipeline_add_spirv_target_passes(pipeline, target_config, backend_config);
    CompilationResult result = shd_pipeline_run(pipeline, compiler_config, module);
    shd_destroy_pipeline(pipeline);

    return result;
}