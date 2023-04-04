#include "runtime_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "list.h"

#include "../common/arena.h"
#include "../common/util.h"

#include "../shady/transform/memory_layout.h"

#include <stdlib.h>

Program* load_program(Runtime* runtime, const char* program_src) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;
    ArenaConfig arena_config = default_arena_config();
    program->arena = new_ir_arena(arena_config);
    CHECK(program->arena != NULL, return false);
    program->module = new_module(program->arena, "my_module");
    CHECK(parse_files(&config, 1, NULL, (const char* []){ program_src }, program->module) == CompilationNoError, return false);
    // TODO split the compilation pipeline into generic and non-generic parts
    append_list(Program*, runtime->programs, program);
    return program;
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    destroy_ir_arena(program->arena);
    free(program);
}

static bool extract_entrypoint_info(const CompilerConfig* config, const Module* mod, EntryPointInfo* info) {
    Nodes decls = get_module_declarations(mod);

    const Node* args_struct_annotation;
    const Node* args_struct_type = NULL;
    const Node* entrypoint = NULL;

    for (int i = 0; i < decls.count; ++i) {
        const Node* node = decls.nodes[i];

        switch (node->tag) {
            case GlobalVariable_TAG: {
                const Node* entry_point_args_annotation = lookup_annotation(node, "EntryPointArgs");
                if (entry_point_args_annotation) {
                    if (node->payload.global_variable.type->tag != RecordType_TAG) {
                        error_print("EntryPointArgs must be a struct\n");
                        return false;
                    }

                    if (args_struct_type) {
                        error_print("there cannot be more than one EntryPointArgs\n");
                        return false;
                    }

                    args_struct_annotation = entry_point_args_annotation;
                    args_struct_type = node->payload.global_variable.type;
                }
                break;
            }
            case Function_TAG: {
                if (lookup_annotation(node, "EntryPoint")) {
                    if (node->payload.fun.params.count != 0) {
                        error_print("EntryPoint cannot have parameters\n");
                        return false;
                    }

                    if (entrypoint) {
                        error_print("there cannot be more than one EntryPoint\n");
                        return false;
                    }

                    entrypoint = node;
                }
                break;
            }
            default: break;
        }
    }

    if (!entrypoint) {
        error_print("could not find EntryPoint\n");
        return false;
    }

    if (!args_struct_type) {
        *info = (EntryPointInfo) { 0 };
        return true;
    }

    if (args_struct_annotation->tag != AnnotationValue_TAG) {
        error_print("EntryPointArgs annotation must contain exactly one value\n");
        return false;
    }
    if (args_struct_annotation->payload.annotation_value.value != entrypoint) {
        error_print("EntryPointArgs annotation refers to different EntryPoint\n");
        return false;
    }

    size_t num_args = args_struct_type->payload.record_type.members.count;

    if (num_args == 0) {
        error_print("EntryPointArgs cannot be empty\n");
        return false;
    }

    IrArena* a = get_module_arena(mod);

    LARRAY(FieldLayout, fields, num_args);
    get_record_layout(config, a, args_struct_type, fields);

    size_t* offset_size_buffer = calloc(1, 2 * num_args * sizeof(size_t));
    if (!offset_size_buffer) {
        error_print("failed to allocate EntryPointArgs offsets and sizes array\n");
        return false;
    }
    size_t* offsets = offset_size_buffer;
    size_t* sizes = offset_size_buffer + num_args;

    for (int i = 0; i < num_args; ++i) {
        offsets[i] = fields[i].offset_in_bytes;
        sizes[i] = fields[i].mem_layout.size_in_bytes;
    }

    info->num_args = num_args;
    info->arg_offset = offsets;
    info->arg_size = sizes;
    info->args_size = offsets[num_args - 1] + sizes[num_args - 1];
    return true;
}

static bool extract_layout(SpecProgram* program) {
    if (program->entrypoint.args_size > program->device->caps.properties.base.properties.limits.maxPushConstantsSize) {
        error_print("EntryPointArgs exceed available push constant space\n");
        return false;
    }

    VkPushConstantRange push_constant_ranges[1] = {
        { .offset = 0, .size = program->entrypoint.args_size, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}
    };

    CHECK_VK(vkCreatePipelineLayout(program->device->device, &(VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pushConstantRangeCount = program->entrypoint.args_size ? sizeof(push_constant_ranges) / sizeof(push_constant_ranges[0]) : 0,
        .pPushConstantRanges = push_constant_ranges,
        .setLayoutCount = 0
    }, NULL, &program->layout), return false);
    return true;
}

static bool create_vk_pipeline(SpecProgram* program) {
    CHECK_VK(vkCreateShaderModule(program->device->device, &(VkShaderModuleCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = program->spirv_size,
        .pCode = (uint32_t*) program->spirv_bytes
    }, NULL, &program->shader_module), return false);

    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .module = program->shader_module,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = program->key.entry_point,
        .pSpecializationInfo = NULL
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info_ext = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = program->device->caps.subgroup_size.max
    };
    if (program->device->caps.supported_extensions[ShadySupportsEXT_subgroup_size_control] &&
       (program->device->caps.properties.subgroup_size_control.requiredSubgroupSizeStages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)) {
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

static CompilerConfig get_compiler_config_for_device(Device* device) {
    CompilerConfig config = default_compiler_config();

    config.subgroup_size = device->caps.subgroup_size.max;
    assert(config.subgroup_size > 0);
    // config.per_thread_stack_size = ...

    config.target_spirv_version.major = device->caps.spirv_version.major;
    config.target_spirv_version.minor = device->caps.spirv_version.minor;

    if (!device->caps.features.subgroup_extended_types.shaderSubgroupExtendedTypes)
        config.lower.emulate_subgroup_ops_extended_types = true;

    config.lower.int64 = !device->caps.features.base.features.shaderInt64;

    if (device->caps.implementation.is_moltenvk) {
        warn_print("Hack: MoltenVK says they supported subgroup extended types, but it's a lie. 64-bit types are unaccounted for !\n");
        config.lower.emulate_subgroup_ops_extended_types = true;
    }
    if (device->caps.properties.base.properties.vendorID == 0x10de) {
        warn_print("Hack: NVidia somehow has unreliable broadcast_first. Emulating it with shuffles seemingly fixes the issue.\n");
        config.hacks.spv_shuffle_instead_of_broadcast_first = true;
    }

    return config;
}

static bool compile_specialized_program(SpecProgram* spec) {
    CompilerConfig config = get_compiler_config_for_device(spec->device);
    config.specialization.entry_point = spec->key.entry_point;

    CHECK(run_compiler_passes(&config, &spec->specialized_module) == CompilationNoError, return false);

    Module* new_mod;
    emit_spirv(&config, spec->specialized_module, &spec->spirv_size, &spec->spirv_bytes, &new_mod);
    spec->specialized_module = new_mod;

    if (spec->key.base->runtime->config.dump_spv) {
        String module_name = get_module_name(spec->specialized_module);
        String file_name = format_string(get_module_arena(spec->specialized_module), "%s.spv", module_name);
        write_file(file_name, spec->spirv_size, (unsigned char*)spec->spirv_bytes);
    }

    return extract_entrypoint_info(&config, spec->specialized_module, &spec->entrypoint);
}

static SpecProgram* create_specialized_program(SpecProgramKey key, Device* device) {
    SpecProgram* spec_program = calloc(1, sizeof(SpecProgram));
    if (!spec_program)
        return NULL;

    spec_program->key = key;
    spec_program->device = device;
    spec_program->specialized_module = key.base->module;

    CHECK(compile_specialized_program(spec_program), return NULL);
    CHECK(extract_layout(spec_program),              return NULL);
    CHECK(create_vk_pipeline(spec_program),          return NULL);
    return spec_program;
}

SpecProgram* get_specialized_program(Program* program, String entry_point, Device* device) {
    SpecProgramKey key = { .base = program, .entry_point = entry_point };
    SpecProgram** found = find_value_dict(SpecProgramKey, SpecProgram*, device->specialized_programs, key);
    if (found)
        return *found;
    SpecProgram* spec = create_specialized_program(key, device);
    assert(spec);
    insert_dict(SpecProgramKey, SpecProgram*, device->specialized_programs, key, spec);
    return spec;
}

void destroy_specialized_program(SpecProgram* spec) {
    vkDestroyPipeline(spec->device->device, spec->pipeline, NULL);
    vkDestroyPipelineLayout(spec->device->device, spec->layout, NULL);
    vkDestroyShaderModule(spec->device->device, spec->shader_module, NULL);
    free(spec->entrypoint.arg_offset);
    free(spec->spirv_bytes);
    if (get_module_arena(spec->specialized_module) != get_module_arena(spec->key.base->module))
        destroy_ir_arena(get_module_arena(spec->specialized_module));
    free(spec);
}
