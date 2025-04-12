#include "vk_runner_private.h"

#include "shady/driver.h"
#include "shady/ir/memory_layout.h"
#include "shady/runtime/runtime.h"
#include "shady/rewrite.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "growy.h"
#include "arena.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

static void register_required_descriptors(VkrSpecProgram* program, VkDescriptorSetLayoutBinding* binding) {
    assert(binding->descriptorCount > 0);
    size_t i = 0;
    while (program->required_descriptor_counts[i].descriptorCount > 0 && program->required_descriptor_counts[i].type != binding->descriptorType) { i++; }
    if (program->required_descriptor_counts[i].descriptorCount == 0) {
        program->required_descriptor_counts[i].type = binding->descriptorType;
        program->required_descriptor_counts_count++;
    }
    program->required_descriptor_counts[i].descriptorCount += binding->descriptorCount;
}

static void add_binding(VkDescriptorSetLayoutCreateInfo* layout_create_info, Growy** bindings_lists, int set, VkDescriptorSetLayoutBinding binding) {
    if (bindings_lists[set] == NULL) {
        bindings_lists[set] = shd_new_growy();
        layout_create_info[set].pBindings = (const VkDescriptorSetLayoutBinding*) shd_growy_data(bindings_lists[set]);
    }
    layout_create_info[set].bindingCount += 1;
    shd_growy_append_object(bindings_lists[set], binding);
}

static bool extract_resources_layout(VkrSpecProgram* program, size_t sets_count, VkDescriptorSetLayout layouts[]) {
    LARRAY(VkDescriptorSetLayoutCreateInfo, layout_create_infos, sets_count);
    LARRAY(Growy*, bindings_lists, sets_count);
    memset(layout_create_infos, 0, sizeof(VkDescriptorSetLayoutCreateInfo) * sets_count);
    memset(bindings_lists, 0, sizeof(Growy*) * sets_count);

    for (size_t i = 0; i < program->interface_items_count; i++) {
        VkrProgramInterfaceItem* item = &program->interface_items[i];
        switch (item->interface_item.dst_kind) {
            case SHD_RII_Dst_Descriptor: break;
            default: continue;
        }

        int set = item->interface_item.dst_details.descriptor.set;
        int binding = item->interface_item.dst_details.descriptor.binding;

        VkDescriptorSetLayoutBinding vk_binding = {
            .binding = binding,
            .descriptorType = item->interface_item.dst_details.descriptor.type,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = NULL,
        };
        register_required_descriptors(program, &vk_binding);
        add_binding(layout_create_infos, bindings_lists, set, vk_binding);
    }

    for (size_t set = 0; set < sets_count; set++) {
        layouts[set] = 0;
        layout_create_infos[set].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_create_infos[set].flags = 0;
        layout_create_infos[set].pNext = NULL;
        vkCreateDescriptorSetLayout(program->device->device, &layout_create_infos[set], NULL, &layouts[set]);
        if (bindings_lists[set] != NULL) {
            shd_destroy_growy(bindings_lists[set]);
        }
    }

    return true;
}

static bool extract_layout(VkrSpecProgram* program) {
    size_t push_constant_size = shd_vkr_get_push_constant_size(program);
    if (push_constant_size > program->device->caps.properties.base.properties.limits.maxPushConstantsSize) {
        shd_error_print("EntryPointArgs exceed available push constant space\n");
        return false;
    }
    VkPushConstantRange push_constant_ranges[1] = {
        { .offset = 0, .size = push_constant_size, .stageFlags = program->stage }
    };

    CHECK(extract_resources_layout(program, MAX_DESCRIPTOR_SETS, program->set_layouts), return false);

    CHECK_VK(vkCreatePipelineLayout(program->device->device, &(VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = push_constant_ranges,
        .setLayoutCount = MAX_DESCRIPTOR_SETS,
        .pSetLayouts = program->set_layouts,
    }, NULL, &program->layout), return false);
    return true;
}

static bool create_vk_shader_module(VkrSpecProgram* spec, String filter_entry_pt, VkShaderModule* out) {
    size_t code_size;
    uint32_t* spirv;

    Module* tmp_mod = NULL, *mod = spec->specialized_module;
    String module_name = shd_module_get_name(spec->specialized_module);
    if (filter_entry_pt) {
        IrArena* a = shd_module_get_arena(mod);
        module_name = shd_fmt_string_irarena(a, "%s_%s", module_name, filter_entry_pt);
        tmp_mod = shd_new_module(a, module_name);

        Rewriter r = shd_create_importer(mod, tmp_mod);
        const Node* ep = shd_module_get_exported(mod, filter_entry_pt);
        assert(ep);
        const Node* new = shd_rewrite_node(&r, ep);
        shd_module_add_export(tmp_mod, shd_get_exported_name(new), new);
        shd_destroy_rewriter(&r);
        mod = tmp_mod;
    }

    shd_emit_spirv(&spec->specialized_config, spec->backend_config, mod, &code_size, (char**) &spirv);

    if (spec->key.base->runtime->config.dump_spv) {
        String file_name = shd_format_string_new("%s.spv", module_name);
        shd_write_file(file_name, code_size, (char*) spirv);
        free((void*) file_name);
    }

    String override_file = getenv("SHADY_OVERRIDE_SPV");
    if (override_file) {
        shd_read_file(override_file, &code_size, (char**) &spirv);
        return true;
    }

    CHECK_VK(vkCreateShaderModule(spec->device->device, &(VkShaderModuleCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = code_size,
        .pCode = spirv
    }, NULL, out), return false);

    free(spirv);
    return true;
}

static bool create_vk_compute_pipeline(VkrSpecProgram* program) {
    create_vk_shader_module(program, NULL, &program->compute.shader_module);

    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .module = program->compute.shader_module,
        .stage = program->stage,
        .pName = program->key.entry_point,
        .pSpecializationInfo = NULL
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info_ext = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = program->device->caps.subgroup_size.max
    };

    if (program->device->caps.supported_extensions[ShadySupportsEXT_subgroup_size_control] &&
       (program->device->caps.properties.subgroup_size_control.requiredSubgroupSizeStages & program->stage)) {
        append_pnext((VkBaseOutStructure*) &stage_create_info, &pipeline_shader_stage_required_subgroup_size_create_info_ext);
    }

    CHECK_VK(vkCreateComputePipelines(program->device->device, VK_NULL_HANDLE, 1, (VkComputePipelineCreateInfo []) { {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .layout = program->layout,
        .stage = stage_create_info,
    } }, NULL, &program->pipeline), return false);
    return true;
}

static int count_callables(VkrSpecProgram* program) {
    const Node* entry_point_decl = shd_module_get_exported(program->specialized_module, program->key.entry_point);
    assert(entry_point_decl);
    const Node* num_callables = shd_lookup_annotation(entry_point_decl, "NumCallables");
    return shd_get_int_value(shd_get_annotation_value(num_callables), false);
}

static bool make_sbt(const VkrSpecProgram* program, size_t base, size_t count, VkrBuffer** p_buffer, VkStridedDeviceAddressRegionKHR* p_region) {
    size_t sbt_handle_size = program->device->caps.properties.rt_pipeline_properties.shaderGroupHandleSize;
    void* sbt_buffer_tmp = NULL;

    if (count) {
        sbt_buffer_tmp = calloc(count, sbt_handle_size);
        size_t sbt_buffer_size = sbt_handle_size * count;
        CHECK_VK(program->device->extensions.vkGetRayTracingShaderGroupHandlesKHR(program->device->device, program->pipeline, base, count, sbt_buffer_size, sbt_buffer_tmp), return false);

        *p_buffer = shd_vkr_allocate_buffer_device(program->device, sbt_buffer_size);
        shd_rn_copy_to_buffer((Buffer*) *p_buffer, 0, sbt_buffer_tmp, sbt_buffer_size);
        free(sbt_buffer_tmp);

        p_region->deviceAddress = shd_rn_get_buffer_device_pointer((Buffer*) *p_buffer);
        p_region->stride = sbt_handle_size;
        p_region->size = sbt_handle_size * count;
    }

    return true;
}

static bool create_vk_rt_pipeline(VkrSpecProgram* program) {
    IrArena* a = shd_module_get_arena(program->specialized_module);
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info_ext = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = program->device->caps.subgroup_size.max
    };

    program->rt.callables_count = count_callables(program);
    LARRAY(VkPipelineShaderStageCreateInfo, stages, 1 + program->rt.callables_count);
    create_vk_shader_module(program, program->key.entry_point, &program->rt.rg_shader_module);
    stages[0] = (VkPipelineShaderStageCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .module = program->rt.rg_shader_module,
        .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        .pName = program->key.entry_point,
        .pSpecializationInfo = NULL
    };

    program->rt.callable_shader_modules = calloc(sizeof(VkShaderModule), program->rt.callables_count);
    for (size_t i = 0; i < program->rt.callables_count; i++) {
        String ep_name = shd_fmt_string_irarena(a, "callee_%d", (int) i);
        create_vk_shader_module(program, ep_name, &program->rt.callable_shader_modules[i]);
        stages[1 + i] = (VkPipelineShaderStageCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .module = program->rt.callable_shader_modules[i],
            .stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
            .pName = ep_name,
            .pSpecializationInfo = NULL
        };
    }

    for (size_t i = 0; i < program->rt.callables_count + 1; i++) {
        if (program->device->caps.supported_extensions[ShadySupportsEXT_subgroup_size_control] &&
            (program->device->caps.properties.subgroup_size_control.requiredSubgroupSizeStages & program->stage)) {
            append_pnext((VkBaseOutStructure*) &stages[i], &pipeline_shader_stage_required_subgroup_size_create_info_ext);
        }
    }

    LARRAY(VkRayTracingShaderGroupCreateInfoKHR, groups, 1 + program->rt.callables_count);
    groups[0] = (VkRayTracingShaderGroupCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
        .generalShader = 0,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .closestHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
    };

    for (size_t i = 0; i < program->rt.callables_count; i++) {
        groups[1 + i] = (VkRayTracingShaderGroupCreateInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = 1 + i,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
    }

    CHECK_VK(program->device->extensions.vkCreateRayTracingPipelinesKHR(program->device->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
        &((VkRayTracingPipelineCreateInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .stageCount = 1 + program->rt.callables_count,
            .pStages = stages,
            .groupCount = 1 + program->rt.callables_count,
            .pGroups = groups,
            .maxPipelineRayRecursionDepth = program->device->caps.properties.rt_pipeline_properties.maxRayRecursionDepth,
            .layout = program->layout,
        }), NULL, &program->pipeline), return false);

    make_sbt(program, 0, 1, &program->rt.rg_sbt_buffer, &program->rt.rg_sbt);
    make_sbt(program, 1, program->rt.callables_count, &program->rt.callables_sbt_buffer, &program->rt.callables_sbt);

    return true;
}

static void get_compiler_config_for_device(VkrDevice* device, CompilerConfig* config, SPVBackendConfig* spv_config) {
    assert(device->caps.subgroup_size.max > 0);
    // config.per_thread_stack_size = ...

    spv_config->target_version.major = device->caps.spirv_version.major;
    spv_config->target_version.minor = device->caps.spirv_version.minor;

    if (!device->caps.features.subgroup_extended_types.shaderSubgroupExtendedTypes)
        config->lower.emulate_subgroup_ops_extended_types = true;

    config->lower.int64 = !device->caps.features.base.features.shaderInt64;

    if (device->caps.implementation.is_moltenvk) {
        shd_warn_print("Hack: MoltenVK says they supported subgroup extended types, but it's a lie. 64-bit types are unaccounted for !\n");
        config->lower.emulate_subgroup_ops_extended_types = true;
        shd_warn_print("Hack: MoltenVK does not support pointers to unsized arrays properly.\n");
        config->lower.decay_ptrs = true;
        spv_config->hacks.avoid_spirv_cross_broken_bda_pointers = true;
    }
    if (device->caps.properties.driver_properties.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        shd_warn_print("Hack: NVidia somehow has unreliable broadcast_first. Emulating it with shuffles seemingly fixes the issue.\n");
        spv_config->hacks.shuffle_instead_of_broadcast_first = true;
    }
}

#include "shady/pipeline/pipeline.h"
#include "shady/pass.h"

void shd_pipeline_add_normalize_input_cf(ShdPipeline pipeline);
void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, TargetConfig tgt);

TargetConfig shd_driver_specialize_target_config(TargetConfig, Module*, ExecutionModel, String);

static bool compile_specialized_program(VkrSpecProgram* spec) {
    spec->specialized_config = *spec->key.base->base_config;
    spec->specialized_module = shd_import(&spec->specialized_config, spec->key.base->module);

    spec->specialized_target = shd_vkr_get_device_target_config(&spec->specialized_config, spec->device);

    spec->backend_config = shd_default_spirv_backend_config();
    get_compiler_config_for_device(spec->device, &spec->specialized_config, &spec->backend_config);

    spec->specialized_target = shd_driver_specialize_target_config(spec->specialized_target, spec->key.base->module, spec->key.em, spec->key.entry_point);

    ShdPipeline pipeline = shd_create_empty_pipeline();
    shd_pipeline_add_normalize_input_cf(pipeline);
    shd_pipeline_add_shader_target_lowering(pipeline, spec->specialized_target);
    shd_pipeline_add_spirv_target_passes(pipeline, &spec->specialized_target, &spec->backend_config);
    CompilationResult result = shd_pipeline_run(pipeline, &spec->specialized_config, &spec->specialized_module);
    shd_destroy_pipeline(pipeline);

    CHECK(result == CompilationNoError, return false);

    shd_vkr_populate_interface(spec);

    return true;
}

static bool allocate_sets(VkrSpecProgram* program) {
    if (program->required_descriptor_counts_count > 0) {
        VkDescriptorPoolCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = MAX_DESCRIPTOR_SETS,
            .pNext = NULL,
            .flags = 0,
            .poolSizeCount = program->required_descriptor_counts_count,
            .pPoolSizes = program->required_descriptor_counts
        };
        CHECK_VK(vkCreateDescriptorPool(program->device->device, &create_info, NULL, &program->descriptor_pool), return false);

        VkDescriptorSetAllocateInfo allocate_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = NULL,
            .pSetLayouts = program->set_layouts,
            .descriptorPool = program->descriptor_pool,
            .descriptorSetCount = MAX_DESCRIPTOR_SETS,
        };
        CHECK_VK(vkAllocateDescriptorSets(program->device->device, &allocate_info, program->sets), return false);
    }

    return true;
}

static bool prepare_program_resources(VkrSpecProgram* program) {
    for (size_t i = 0; i < program->interface_items_count; i++) {
        VkrProgramInterfaceItem* item = &program->interface_items[i];

        switch (item->interface_item.src_kind) {
            case SHD_RII_Src_Param:
                continue;
            case SHD_RII_Src_LiftedConstant: {
                size_t size;
                shd_rt_materialize_constant(item->interface_item.src_details.lifted_constant.constant, &size, NULL);

                if (shd_vkr_can_import_host_memory(program->device)) {
                    item->host_owning_ptr = shd_alloc_aligned(size, program->device->caps.properties.external_memory_host.minImportedHostPointerAlignment);
                    item->buffer = shd_vkr_import_buffer_host(program->device, item->host_owning_ptr, size);
                } else {
                    item->buffer = shd_vkr_allocate_buffer_device(program->device, size);
                }

                void* materialized_constant_data = calloc(size, 1);
                shd_rt_materialize_constant(item->interface_item.src_details.lifted_constant.constant, &size, materialized_constant_data);
                shd_rn_copy_to_buffer((Buffer*) item->buffer, 0, materialized_constant_data, size);

                break;
            }
            case SHD_RII_Src_ScratchBuffer: {
                // the exact computation has to happen at launch time
                item->per_invocation_size = shd_get_int_value(item->interface_item.src_details.scratch_buffer.per_invocation_size, false);
                break;
            }
        }
    }

    return true;
}

static VkrSpecProgram* create_specialized_program(SpecProgramKey key, VkrDevice* device) {
    VkrSpecProgram* spec_program = calloc(1, sizeof(VkrSpecProgram));
    if (!spec_program)
        return NULL;

    spec_program->key = key;
    spec_program->device = device;
    spec_program->arena = shd_new_arena();

    switch (key.em) {
        case EmCompute:
            spec_program->stage = VK_SHADER_STAGE_COMPUTE_BIT;
            spec_program->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            break;
        case EmRayGeneration:
            spec_program->stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            spec_program->bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
            break;
        default: shd_error("Unsupported stage")
    }

    CHECK(compile_specialized_program(spec_program), return NULL);
    CHECK(extract_layout(spec_program),              return NULL);
    switch (spec_program->stage) {
        case VK_SHADER_STAGE_COMPUTE_BIT:
            CHECK(create_vk_compute_pipeline(spec_program), return NULL);
            break;
        case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
            CHECK(create_vk_rt_pipeline(spec_program), return NULL);
            break;
        default: shd_error("")
    }
    CHECK(allocate_sets(spec_program),               return NULL);
    CHECK(prepare_program_resources(spec_program),   return NULL);
    return spec_program;
}

VkrSpecProgram* shd_vkr_get_specialized_program(Program* program, String entry_point, VkrDevice* device) {
    const Node* entry_point_decl = shd_module_get_exported(program->module, entry_point);
    assert(entry_point_decl);

    const Node* entry_point_annotation = shd_lookup_annotation(entry_point_decl, "EntryPoint");
    ExecutionModel execution_model = shd_execution_model_from_string(shd_get_string_literal(entry_point_annotation->arena, shd_get_annotation_value(entry_point_annotation)));

    SpecProgramKey key = { .base = program, .entry_point = entry_point, .em = execution_model };
    VkrSpecProgram** found = shd_dict_find_value(SpecProgramKey, VkrSpecProgram*, device->specialized_programs, key);
    if (found)
        return *found;
    VkrSpecProgram* spec = create_specialized_program(key, device);
    assert(spec);
    shd_dict_insert(SpecProgramKey, VkrSpecProgram*, device->specialized_programs, key, spec);
    return spec;
}

void shd_vkr_destroy_specialized_program(VkrSpecProgram* spec) {
    vkDestroyPipeline(spec->device->device, spec->pipeline, NULL);
    for (size_t set = 0; set < MAX_DESCRIPTOR_SETS; set++)
        vkDestroyDescriptorSetLayout(spec->device->device, spec->set_layouts[set], NULL);
    vkDestroyPipelineLayout(spec->device->device, spec->layout, NULL);
    //vkDestroyShaderModule(spec->device->device, spec->shader_module, NULL);
    //free(spec->spirv_bytes);
    if (shd_module_get_arena(spec->specialized_module) != shd_module_get_arena(spec->key.base->module))
        shd_destroy_ir_arena(shd_module_get_arena(spec->specialized_module));
    for (size_t i = 0; i < spec->interface_items_count; i++) {
        VkrProgramInterfaceItem* item = &spec->interface_items[i];
        if (item->buffer)
            shd_vkr_destroy_buffer(item->buffer);
        if (item->host_owning_ptr)
            shd_free_aligned(item->host_owning_ptr);
    }
    free(spec->interface_items);
    vkDestroyDescriptorPool(spec->device->device, spec->descriptor_pool, NULL);
    shd_destroy_arena(spec->arena);
    free(spec);
}
