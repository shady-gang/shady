#include "shady/driver.h"
#include "shady/ir.h"

#include "util.h"
#include "log.h"

#include <stdlib.h>

static void add_logical_ptr_limitations(TargetConfig* target_config) {
    // by default everything is logical
    for (int i = 0; i < NumAddressSpaces; i++)
        target_config->memory.address_spaces[(AddressSpace) i].physical = false;
    // no generic pointers either
    target_config->memory.address_spaces[AsGeneric].allowed = false;
}

static void add_default_shading_language_limitations(TargetConfig* target_config) {
    add_logical_ptr_limitations(target_config);
    target_config->capabilities.native_fncalls = false;
    target_config->capabilities.native_tailcalls = false;
    target_config->capabilities.native_memcpy = false;
    target_config->capabilities.native_stack = false;
    target_config->capabilities.linkage = false;
}

static CodegenTarget guess_target_through_name(const char* filename) {
    if (shd_string_ends_with(filename, ".c"))
        return TgtC;
    else if (shd_string_ends_with(filename, "glsl"))
        return TgtGLSL;
    else if (shd_string_ends_with(filename, "spirv") || shd_string_ends_with(filename, "spv"))
        return TgtSPV;
    else if (shd_string_ends_with(filename, "ispc"))
        return TgtISPC;
    shd_error_print("No target has been specified, and output filename '%s' did not allow guessing the right one\n");
    exit(ShdInvalidTarget);
}

static void configure_target(TargetConfig* target_config, CodegenTarget target_type, const CompilerConfig* compiler_config, DriverConfig* driver_config) {
    if (target_type == TgtNone) {
        if (driver_config && driver_config->output_filename) {
            target_type = driver_config->target_type = guess_target_through_name(driver_config->output_filename);
        } else {
            shd_log_fmt(INFO, "No target specified, defaulting to a generic one.\n");
        }
    }

    switch (target_type) {
        case TgtNone: /* no target */  break;
        case TgtSPV:
            if (driver_config)
                driver_config->backend_type = BackendSPV;
            // Default to 64-wide to support GCN cards.
            target_config->subgroup_size = 64;
            add_default_shading_language_limitations(target_config);
            // default to assuming BDA support on Vulkan
            target_config->memory.address_spaces[AsGlobal].physical = true;

            if (compiler_config->use_rt_pipelines_for_calls) {
                target_config->capabilities.native_fncalls = true;
                target_config->capabilities.rt_pipelines = true;
                target_config->memory.fn_ptr_size = ShdIntSize32;
            }
            break;
        case TgtC:
            if (driver_config) {
                driver_config->backend_type = BackendC;
                driver_config->backend_config.c.dialect = CDialect_C11;
            }
            break;
        case TgtGLSL:
            if (driver_config) {
                driver_config->backend_type = BackendC;
                driver_config->backend_config.c.dialect = CDialect_GLSL;
            }
            add_default_shading_language_limitations(target_config);
            break;
        case TgtISPC:
            if (driver_config) {
                driver_config->backend_type = BackendC;
                driver_config->backend_config.c.dialect = CDialect_ISPC;
            }
            target_config->subgroup_size = 8;
            add_default_shading_language_limitations(target_config);
            for (size_t i = 0; i < NumAddressSpaces; i++) {
                if (i != AsGeneric && shd_get_addr_space_scope(i) < ShdScopeSubgroup) {
                    // ISPC can use native physical pointers for `uniform` data
                    // Due to how it lays out types for `varying` data, we want to emulate memory ourselves.
                    target_config->memory.address_spaces[AsGlobal].physical = true;
                }
            }
            break;
        case TgtCUDA:
            if (driver_config) {
                driver_config->backend_type = BackendC;
                driver_config->backend_config.c.dialect = CDialect_CUDA;
            }
            target_config->subgroup_size = 32;
            //target_config->memory.fn_ptr_size = IntTy64;
            //target_config->memory.word_size = IntTy8;
            //target_config->capabilities.native_stack = true;
            //target_config->capabilities.native_memcpy = true;
            //target_config->memory.max_align = 8;
            add_default_shading_language_limitations(target_config);
            target_config->capabilities.linkage = true;
            target_config->memory.address_spaces[AsGlobal].physical = true;
            break;
    }
}

void shd_driver_configure_defaults_for_target(TargetConfig* target_config, const CompilerConfig* compiler_config, CodegenTarget target) {
    configure_target(target_config, target, compiler_config, NULL);
}

void shd_driver_configure_target(TargetConfig* target_config, DriverConfig* driver_config) {
    configure_target(target_config, driver_config->target_type, &driver_config->config, driver_config);
}
