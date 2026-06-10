#ifndef SHD_JIT_VULKAN
#define SHD_JIT_VULKAN

#include "shady/runtime/vulkan.h"
#include "shady/pipeline/shader_pipeline.h"
#include "shady/be/spirv.h"

/// Configures the SPIR-V backend for a particular target device
void shd_jit_vk_get_compiler_config_for_device(const ShadyVkrPhysicalDeviceCaps* caps, const TargetConfig* target_config, SPVBackendConfig* spv_config, /* TODO: remove */ CompilerConfig* config);

/// Produces a
ShdResult shd_jit_vk_compile_module(Module** module, const TargetConfig* target_config, const ShaderLoweringConfig*, const SPVBackendConfig* backend_config, const CompilerConfig* compiler_config);

#endif
