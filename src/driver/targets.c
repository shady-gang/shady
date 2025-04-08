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
    exit(InvalidTarget);
}

void shd_driver_configure_target(DriverConfig* driver_config) {
    TargetConfig* target_config = &driver_config->target;

    if (driver_config->target_type == TgtAuto) {
        if (driver_config->output_filename) {
            driver_config->target_type = guess_target_through_name(driver_config->output_filename);
        } else {
            shd_log_fmt(INFO, "No target specified, defaulting to a generic one.\n");
        }
    }

    switch (driver_config->target_type) {
        case TgtAuto: /* no target */  break;
        case TgtSPV:
            driver_config->backend_type = BackendSPV;
            add_default_shading_language_limitations(target_config);
            // default to assuming BDA support on Vulkan
            target_config->memory.address_spaces[AsGlobal].physical = true;
            break;
        case TgtC:
            driver_config->backend_type = BackendC;
            driver_config->backend_config.c.dialect = CDialect_C11;
            break;
        case TgtGLSL:
            driver_config->backend_type = BackendC;
            driver_config->backend_config.c.dialect = CDialect_GLSL;
            add_default_shading_language_limitations(target_config);
            break;
        case TgtISPC:
            driver_config->backend_type = BackendC;
            driver_config->backend_config.c.dialect = CDialect_ISPC;
            add_default_shading_language_limitations(target_config);
            break;
    }
}

ExecutionModel shd_execution_model_from_entry_point(const Node* decl) {
    String name = shd_get_node_name_safe(decl);
    if (decl->tag != Function_TAG)
        shd_error("Cannot specialize: '%s' is not a function.", name)
    const Node* ep = shd_lookup_annotation(decl, "EntryPoint");
    if (!ep)
        shd_error("%s is not annotated with @EntryPoint", name);
    return shd_execution_model_from_string(shd_get_annotation_string_payload(ep));
}

static ExecutionModel get_execution_model_for_entry_point(String entry_point, const Module* mod) {
    const Node* decl = shd_module_get_exported(mod, entry_point);
    if (!decl)
        shd_error("Cannot specialize: No function named '%s'", entry_point)
    return shd_execution_model_from_entry_point(decl);
}

void shd_pipeline_add_normalize_input_cf(ShdPipeline);
void shd_pipeline_add_shader_target_lowering(ShdPipeline, TargetConfig tgt, ExecutionModel em, String entry_point);
void shd_pipeline_add_feature_lowering(ShdPipeline, TargetConfig);

void shd_driver_fill_pipeline(ShdPipeline pipeline, const DriverConfig* driver_config, Module* mod) {
    shd_pipeline_add_normalize_input_cf(pipeline);

    // specialize the target config a bit further based on the module
    TargetConfig specialized_target_config = driver_config->target;
    ExecutionModel execution_model = driver_config->specialization.execution_model;
    if (driver_config->specialization.entry_point && execution_model == EmNone) {
        execution_model = get_execution_model_for_entry_point(driver_config->specialization.entry_point, mod);
    }
    if (execution_model != EmNone)
        specialized_target_config.execution_model = execution_model;

    shd_pipeline_add_shader_target_lowering(pipeline, specialized_target_config, execution_model, driver_config->specialization.entry_point);

    switch (driver_config->backend_type) {
        case BackendNone: /* do nothing */ break;
        case BackendC:
            shd_pipeline_add_c_target_passes(pipeline, &driver_config->backend_config.c);
            break;
        case BackendSPV:
            shd_pipeline_add_spirv_target_passes(pipeline, &driver_config->backend_config.spirv);
            break;
    }
}