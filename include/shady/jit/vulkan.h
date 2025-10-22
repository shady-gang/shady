#ifndef SHD_JIT_VULKAN
#define SHD_JIT_VULKAN

#include "shady/runtime/vulkan.h"
#include "shady/driver.h"

/// Configures the SPIR-V backend for a particular target device
void shd_jit_vk_get_compiler_config_for_device(const ShadyVkrPhysicalDeviceCaps* caps, const TargetConfig* target_config, SPVBackendConfig* spv_config, /* TODO: remove */ CompilerConfig* config);

/// Produces a
CompilationResult shd_jit_vk_compile_module(Module** module, const TargetConfig* target_config, const SPVBackendConfig* backend_config, const CompilerConfig* compiler_config);

#endif
