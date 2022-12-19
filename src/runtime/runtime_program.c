#include "runtime_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"
#include "list.h"

#include <stdlib.h>
#include <string.h>

Program* load_program(Runtime* runtime, const char* program_src) {
    Program* program = malloc(sizeof(Program));
    memset(program, 0, sizeof(Program));
    program->runtime = runtime;

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;
    ArenaConfig arena_config = { 0 };
    program->arena = new_ir_arena(arena_config);
    CHECK(program->arena != NULL, return false);
    program->generic_program = new_module(program->arena, "my_module");
    CHECK(parse_files(&config, 1, (const char* []){ program_src }, program->generic_program) == CompilationNoError, return false);
    // TODO split the compilation pipeline into generic and non-generic parts
    append_list(Program*, runtime->programs, program);
    return program;
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    destroy_ir_arena(program->arena);
    free(program);
}

static bool extract_layout(SpecProgram * program) {
    CHECK_VK(vkCreatePipelineLayout(program->device->device, &(VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pushConstantRangeCount = 0, // TODO !
        .setLayoutCount = 0 // TODO !
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

    CHECK_VK(vkCreateComputePipelines(program->device->device, VK_NULL_HANDLE, 1, (VkComputePipelineCreateInfo []) { {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .layout = program->layout,
        .stage = (VkPipelineShaderStageCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .module = program->shader_module,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
            .pSpecializationInfo = NULL
        }
    } }, NULL, &program->pipeline), return false);
    return true;
}

static CompilerConfig get_compiler_config_for_device(Device* device) {
    CompilerConfig config = default_compiler_config();

    config.subgroup_size = device->properties.subgroup_size;
    assert(config.subgroup_size > 0);
    // config.per_thread_stack_size = ...

    config.target_spirv_version.major = device->properties.spirv_version.major;
    config.target_spirv_version.minor = device->properties.spirv_version.minor;

    if (!device->properties.features.subgroup_extended_types)
        config.lower.emulate_subgroup_ops_extended_types = true;

    if (device->properties.implementation.is_moltenvk)
        config.lower.emulate_subgroup_ops_extended_types = true;

    config.logging.skip_generated = true;
    config.logging.skip_builtin = true;

    return config;
}

static bool compile_specialized_program(SpecProgram* spec) {
    CompilerConfig config = get_compiler_config_for_device(spec->device);

    CHECK(run_compiler_passes(&config, &spec->module) == CompilationNoError, return false);
    emit_spirv(&config, spec->module, &spec->spirv_size, &spec->spirv_bytes);
    if (spec->base->runtime->config.dump_spv) {
        FILE* f = fopen("runtime-dump.spv", "wb");
        fwrite(spec->spirv_bytes, 1, spec->spirv_size, f);
        fclose(f);
    }
    return true;
}

static SpecProgram* create_specialized_program(Program* program, Device* device) {
    SpecProgram* spec_program = calloc(1, sizeof(SpecProgram));
    spec_program->base = program;
    spec_program->device = device;

    ArenaConfig arena_config = { 0 };
    spec_program->arena = new_ir_arena(arena_config);
    spec_program->module = program->generic_program;

    CHECK(compile_specialized_program(spec_program), return NULL);
    CHECK(extract_layout(spec_program),              return NULL);
    CHECK(create_vk_pipeline(spec_program),          return NULL);
    return spec_program;
}

SpecProgram* get_specialized_program(Program* program, Device* device) {
    SpecProgram** found = find_value_dict(Program*, SpecProgram*, device->specialized_programs, program);
    if (found)
        return *found;
    SpecProgram* spec = create_specialized_program(program, device);
    assert(spec);
    insert_dict(Program*, SpecProgram*, device->specialized_programs, program, spec);
    return spec;
}

void destroy_specialized_program(SpecProgram* spec) {
    vkDestroyPipeline(spec->device->device, spec->pipeline, NULL);
    vkDestroyPipelineLayout(spec->device->device, spec->layout, NULL);
    vkDestroyShaderModule(spec->device->device, spec->shader_module, NULL);
    free(spec->spirv_bytes);
    destroy_ir_arena(spec->arena);
    assert(spec->arena != get_module_arena(spec->module));
    destroy_ir_arena(get_module_arena(spec->module));
    free(spec);
}