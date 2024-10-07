#include "vk_runtime_private.h"

#include "shady/driver.h"
#include "shady/memory_layout.h"

#include "type.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "list.h"
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

VkDescriptorType as_to_descriptor_type(AddressSpace as) {
    switch (as) {
        case AsUniform: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case AsShaderStorageBufferObject: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        default: shd_error("No mapping to a descriptor type");
    }
}

static void write_value(unsigned char* tgt, const Node* value) {
    IrArena* a = value->arena;
    switch (value->tag) {
        case IntLiteral_TAG: {
            switch (value->payload.int_literal.width) {
                case IntTy8: *((uint8_t*) tgt) = (uint8_t) (value->payload.int_literal.value & 0xFF); break;
                case IntTy16: *((uint16_t*) tgt) = (uint16_t) (value->payload.int_literal.value & 0xFFFF); break;
                case IntTy32: *((uint32_t*) tgt) = (uint32_t) (value->payload.int_literal.value & 0xFFFFFFFF); break;
                case IntTy64: *((uint64_t*) tgt) = (uint64_t) (value->payload.int_literal.value); break;
            }
            break;
        }
        case Composite_TAG: {
            Nodes values = value->payload.composite.contents;
            const Type* struct_t = value->payload.composite.type;
            struct_t = get_maybe_nominal_type_body(struct_t);

            if (struct_t->tag == RecordType_TAG) {
                LARRAY(FieldLayout, fields, values.count);
                shd_get_record_layout(a, struct_t, fields);
                for (size_t i = 0; i < values.count; i++) {
                    // TypeMemLayout layout = get_mem_layout(value->arena, get_unqualified_type(element->type));
                    write_value(tgt + fields->offset_in_bytes, values.nodes[i]);
                }
            } else if (struct_t->tag == ArrType_TAG) {
                for (size_t i = 0; i < values.count; i++) {
                    TypeMemLayout layout = shd_get_mem_layout(value->arena, get_unqualified_type(values.nodes[i]->type));
                    write_value(tgt, values.nodes[i]);
                    tgt += layout.size_in_bytes;
                }
            } else {
                assert(false);
            }
            break;
        }
        default:
            assert(false);
    }
}

static bool extract_resources_layout(VkrSpecProgram* program, VkDescriptorSetLayout layouts[]) {
    VkDescriptorSetLayoutCreateInfo layout_create_infos[MAX_DESCRIPTOR_SETS] = { 0 };
    Growy* bindings_lists[MAX_DESCRIPTOR_SETS] = { 0 };
    Growy* resources = shd_new_growy();

    Nodes decls = get_module_declarations(program->specialized_module);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;

        if (lookup_annotation(decl, "Constants")) {
            AddressSpace as = decl->payload.global_variable.address_space;
            switch (as) {
                case AsShaderStorageBufferObject:
                case AsUniform: break;
                default: continue;
            }

            int set = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(lookup_annotation(decl, "DescriptorSet"))), false);
            int binding = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(lookup_annotation(decl, "DescriptorBinding"))), false);

            ProgramResourceInfo* res_info = shd_arena_alloc(program->arena, sizeof(ProgramResourceInfo));
            *res_info = (ProgramResourceInfo) {
                .is_bound = true,
                .as = as,
                .set = set,
                .binding = binding,
            };
            shd_growy_append_object(resources, res_info);
            program->resources.num_resources++;

            const Type* struct_t = decl->payload.global_variable.type;
            assert(struct_t->tag == RecordType_TAG && struct_t->payload.record_type.special == DecorateBlock);

            for (size_t j = 0; j < struct_t->payload.record_type.members.count; j++) {
                const Type* member_t = struct_t->payload.record_type.members.nodes[j];
                assert(member_t->tag == PtrType_TAG);
                member_t = get_pointee_type(member_t->arena, member_t);
                TypeMemLayout layout = shd_get_mem_layout(get_module_arena(program->specialized_module), member_t);

                ProgramResourceInfo* constant_res_info = shd_arena_alloc(program->arena, sizeof(ProgramResourceInfo));
                *constant_res_info = (ProgramResourceInfo) {
                    .parent = res_info,
                    .as = as,
                };
                shd_growy_append_object(resources, constant_res_info);
                program->resources.num_resources++;

                constant_res_info->size = layout.size_in_bytes;
                constant_res_info->offset = res_info->size;
                res_info->size += sizeof(void*);

                // TODO initial value
                Nodes annotations = get_declaration_annotations(decl);
                for (size_t k = 0; k < annotations.count; k++) {
                    const Node* a = annotations.nodes[k];
                    if ((strcmp(get_annotation_name(a), "InitialValue") == 0) && resolve_to_int_literal(shd_first(get_annotation_values(a)))->value == j) {
                        constant_res_info->default_data = calloc(1, layout.size_in_bytes);
                        write_value(constant_res_info->default_data, get_annotation_values(a).nodes[1]);
                        //printf("wowie");
                    }
                }
            }

            if (vkr_can_import_host_memory(program->device))
                res_info->host_backed_allocation = true;
            else
                res_info->staging = calloc(1, res_info->size);

            VkDescriptorSetLayoutBinding vk_binding = {
                .binding = binding,
                .descriptorType = as_to_descriptor_type(as),
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = NULL,
            };
            register_required_descriptors(program, &vk_binding);
            add_binding(layout_create_infos, bindings_lists, set, vk_binding);
        }
    }

    for (size_t set = 0; set < MAX_DESCRIPTOR_SETS; set++) {
        layouts[set] = 0;
        layout_create_infos[set].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_create_infos[set].flags = 0;
        layout_create_infos[set].pNext = NULL;
        vkCreateDescriptorSetLayout(program->device->device, &layout_create_infos[set], NULL, &layouts[set]);
        if (bindings_lists[set] != NULL) {
            shd_destroy_growy(bindings_lists[set]);
        }
    }

    program->resources.resources = (ProgramResourceInfo**) shd_growy_deconstruct(resources);

    return true;
}

static bool extract_layout(VkrSpecProgram* program) {
    if (program->parameters.args_size > program->device->caps.properties.base.properties.limits.maxPushConstantsSize) {
        shd_error_print("EntryPointArgs exceed available push constant space\n");
        return false;
    }
    VkPushConstantRange push_constant_ranges[1] = {
        { .offset = 0, .size = program->parameters.args_size, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}
    };

    CHECK(extract_resources_layout(program, program->set_layouts), return false);

    CHECK_VK(vkCreatePipelineLayout(program->device->device, &(VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pushConstantRangeCount = program->parameters.args_size ? sizeof(push_constant_ranges) / sizeof(push_constant_ranges[0]) : 0,
        .pPushConstantRanges = push_constant_ranges,
        .setLayoutCount = MAX_DESCRIPTOR_SETS,
        .pSetLayouts = program->set_layouts,
    }, NULL, &program->layout), return false);
    return true;
}

static bool create_vk_pipeline(VkrSpecProgram* program) {
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
       (program->device->caps.properties.subgroup_size_control.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
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

static CompilerConfig get_compiler_config_for_device(VkrDevice* device, const CompilerConfig* base_config) {
    CompilerConfig config = *base_config;

    assert(device->caps.subgroup_size.max > 0);
    config.specialization.subgroup_size = device->caps.subgroup_size.max;
    // config.per_thread_stack_size = ...

    config.target_spirv_version.major = device->caps.spirv_version.major;
    config.target_spirv_version.minor = device->caps.spirv_version.minor;

    if (!device->caps.features.subgroup_extended_types.shaderSubgroupExtendedTypes)
        config.lower.emulate_subgroup_ops_extended_types = true;

    config.lower.int64 = !device->caps.features.base.features.shaderInt64;

    if (device->caps.implementation.is_moltenvk) {
        shd_warn_print("Hack: MoltenVK says they supported subgroup extended types, but it's a lie. 64-bit types are unaccounted for !\n");
        config.lower.emulate_subgroup_ops_extended_types = true;
        shd_warn_print("Hack: MoltenVK does not support pointers to unsized arrays properly.\n");
        config.lower.decay_ptrs = true;
    }
    if (device->caps.properties.driver_properties.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        shd_warn_print("Hack: NVidia somehow has unreliable broadcast_first. Emulating it with shuffles seemingly fixes the issue.\n");
        config.hacks.spv_shuffle_instead_of_broadcast_first = true;
    }

    return config;
}

static bool extract_parameters_info(ProgramParamsInfo* parameters, Module* mod) {
    Nodes decls = get_module_declarations(mod);

    const Node* args_struct_annotation;
    const Node* args_struct_type = NULL;
    const Node* entry_point_function = NULL;

    for (int i = 0; i < decls.count; ++i) {
        const Node* node = decls.nodes[i];

        switch (node->tag) {
            case GlobalVariable_TAG: {
                const Node* entry_point_args_annotation = lookup_annotation(node, "EntryPointArgs");
                if (entry_point_args_annotation) {
                    if (node->payload.global_variable.type->tag != RecordType_TAG) {
                        shd_error_print("EntryPointArgs must be a struct\n");
                        return false;
                    }

                    if (args_struct_type) {
                        shd_error_print("there cannot be more than one EntryPointArgs\n");
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
                        shd_error_print("EntryPoint cannot have parameters\n");
                        return false;
                    }

                    if (entry_point_function) {
                        shd_error_print("there cannot be more than one EntryPoint\n");
                        return false;
                    }

                    entry_point_function = node;
                }
                break;
            }
            default: break;
        }
    }

    if (!entry_point_function) {
        shd_error_print("could not find EntryPoint\n");
        return false;
    }

    if (!args_struct_type) {
        *parameters = (ProgramParamsInfo) { .num_args = 0 };
        return true;
    }

    if (args_struct_annotation->tag != AnnotationValue_TAG) {
        shd_error_print("EntryPointArgs annotation must contain exactly one value\n");
        return false;
    }

    const Node* annotation_fn = args_struct_annotation->payload.annotation_value.value;
    assert(annotation_fn->tag == FnAddr_TAG);
    if (annotation_fn->payload.fn_addr.fn != entry_point_function) {
        shd_error_print("EntryPointArgs annotation refers to different EntryPoint\n");
        return false;
    }

    size_t num_args = args_struct_type->payload.record_type.members.count;

    if (num_args == 0) {
        shd_error_print("EntryPointArgs cannot be shd_empty\n");
        return false;
    }

    IrArena* a = get_module_arena(mod);

    LARRAY(FieldLayout, fields, num_args);
    shd_get_record_layout(a, args_struct_type, fields);

    size_t* offset_size_buffer = calloc(1, 2 * num_args * sizeof(size_t));
    if (!offset_size_buffer) {
        shd_error_print("failed to allocate EntryPointArgs offsets and sizes array\n");
        return false;
    }
    size_t* offsets = offset_size_buffer;
    size_t* sizes = offset_size_buffer + num_args;

    for (int i = 0; i < num_args; ++i) {
        offsets[i] = fields[i].offset_in_bytes;
        sizes[i] = fields[i].mem_layout.size_in_bytes;
    }

    parameters->num_args = num_args;
    parameters->arg_offset = offsets;
    parameters->arg_size = sizes;
    parameters->args_size = offsets[num_args - 1] + sizes[num_args - 1];
    return true;
}

static bool compile_specialized_program(VkrSpecProgram* spec) {
    CompilerConfig config = get_compiler_config_for_device(spec->device, spec->key.base->base_config);
    config.specialization.entry_point = spec->key.entry_point;

    CHECK(shd_run_compiler_passes(&config, &spec->specialized_module) == CompilationNoError, return false);

    Module* final_mod;
    emit_spirv(&config, spec->specialized_module, &spec->spirv_size, &spec->spirv_bytes, &final_mod);

    CHECK(extract_parameters_info(&spec->parameters, final_mod), return false);

    spec->specialized_module = final_mod;

    if (spec->key.base->runtime->config.dump_spv) {
        String module_name = get_module_name(spec->specialized_module);
        String file_name = shd_format_string_new("%s.spv", module_name);
        shd_write_file(file_name, spec->spirv_size, (const char*) spec->spirv_bytes);
        free((void*) file_name);
    }

    String override_file = getenv("SHADY_OVERRIDE_SPV");
    if (override_file) {
        shd_read_file(override_file, &spec->spirv_size, &spec->spirv_bytes);
        return true;
    }

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

static void flush_staged_data(VkrSpecProgram* program) {
    for (size_t i = 0; i < program->resources.num_resources; i++) {
        ProgramResourceInfo* resource = program->resources.resources[i];
        if (resource->staging) {
            copy_to_buffer((Buffer*) resource->buffer, 0, resource->buffer, resource->size);
            free(resource->staging);
        }
    }
}

static bool prepare_resources(VkrSpecProgram* program) {
    for (size_t i = 0; i < program->resources.num_resources; i++) {
        ProgramResourceInfo* resource = program->resources.resources[i];

        if (resource->host_backed_allocation) {
            assert(vkr_can_import_host_memory(program->device));
            resource->host_ptr = shd_alloc_aligned(resource->size, program->device->caps.properties.external_memory_host.minImportedHostPointerAlignment);
            resource->buffer = vkr_import_buffer_host(program->device, resource->host_ptr, resource->size);
        } else {
            resource->buffer = vkr_allocate_buffer_device(program->device, resource->size);
        }

        if (resource->default_data) {
            copy_to_buffer((Buffer*) resource->buffer, 0, resource->default_data, resource->size);
        } else {
            char* zeroes = calloc(1, resource->size);
            copy_to_buffer((Buffer*) resource->buffer, 0, zeroes, resource->size);
            free(zeroes);
        }

        if (resource->parent) {
            char* dst = resource->parent->host_ptr;
            if (!dst) {
                dst = resource->parent->staging;
            }
            assert(dst);
            *((uint64_t*) (dst + resource->offset)) = get_buffer_device_pointer((Buffer*) resource->buffer);
        }
    }

    flush_staged_data(program);

    return true;
}

static VkrSpecProgram* create_specialized_program(SpecProgramKey key, VkrDevice* device) {
    VkrSpecProgram* spec_program = calloc(1, sizeof(VkrSpecProgram));
    if (!spec_program)
        return NULL;

    spec_program->key = key;
    spec_program->device = device;
    spec_program->specialized_module = key.base->module;
    spec_program->arena = shd_new_arena();

    CHECK(compile_specialized_program(spec_program), return NULL);
    CHECK(extract_layout(spec_program),              return NULL);
    CHECK(create_vk_pipeline(spec_program),          return NULL);
    CHECK(allocate_sets(spec_program),               return NULL);
    CHECK(prepare_resources(spec_program),           return NULL);
    return spec_program;
}

VkrSpecProgram* get_specialized_program(Program* program, String entry_point, VkrDevice* device) {
    SpecProgramKey key = { .base = program, .entry_point = entry_point };
    VkrSpecProgram** found = shd_dict_find_value(SpecProgramKey, VkrSpecProgram*, device->specialized_programs, key);
    if (found)
        return *found;
    VkrSpecProgram* spec = create_specialized_program(key, device);
    assert(spec);
    shd_dict_insert(SpecProgramKey, VkrSpecProgram*, device->specialized_programs, key, spec);
    return spec;
}

void destroy_specialized_program(VkrSpecProgram* spec) {
    vkDestroyPipeline(spec->device->device, spec->pipeline, NULL);
    for (size_t set = 0; set < MAX_DESCRIPTOR_SETS; set++)
        vkDestroyDescriptorSetLayout(spec->device->device, spec->set_layouts[set], NULL);
    vkDestroyPipelineLayout(spec->device->device, spec->layout, NULL);
    vkDestroyShaderModule(spec->device->device, spec->shader_module, NULL);
    free( (void*) spec->parameters.arg_offset);
    free(spec->spirv_bytes);
    if (get_module_arena(spec->specialized_module) != get_module_arena(spec->key.base->module))
        shd_destroy_ir_arena(get_module_arena(spec->specialized_module));
    for (size_t i = 0; i < spec->resources.num_resources; i++) {
        ProgramResourceInfo* resource = spec->resources.resources[i];
        if (resource->buffer)
            vkr_destroy_buffer(resource->buffer);
        if (resource->host_ptr && resource->host_backed_allocation)
            shd_free_aligned(resource->host_ptr);
    }
    free(spec->resources.resources);
    vkDestroyDescriptorPool(spec->device->device, spec->descriptor_pool, NULL);
    shd_destroy_arena(spec->arena);
    free(spec);
}
