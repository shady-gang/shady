#include "runtime_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <stdlib.h>
#include <string.h>

#include "murmur3.h"

struct Program_ {
    Runtime* runtime;

    IrArena* arena;
    const Node* generic_program;

    struct Dict* specialized;
};

struct SpecProgram_ {
    Program* base;
    Device* device;

    IrArena* arena;
    const Node* final_program;

    size_t spirv_size;
    char* spirv_bytes;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;
};

// TODO: unduplicate
static KeyHash hash_murmur(const void* data, size_t size) {
    int32_t out[4];
    MurmurHash3_x64_128(data, (int) size, 0x1234567, &out);

    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

KeyHash hash_device(Device** pdevice) {
    return hash_murmur(*pdevice, sizeof(Device*));
}

bool cmp_devices(Device** pldevice, Device** prdevice) {
    return *pldevice == *prdevice;
}

Program* load_program(Runtime* runtime, const char* program_src) {
    Program* program = malloc(sizeof(Program));
    memset(program, 0, sizeof(Program));
    program->runtime = runtime;

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;
    ArenaConfig arena_config = {};
    program->arena = new_arena(arena_config);
    CHECK(program->arena != NULL, return false);
    CHECK(parse_files(&config, 1, (const char* []){ program_src }, program->arena, &program->generic_program) == CompilationNoError, return false);
    // TODO split the compilation pipeline into generic and non-generic parts

    program->specialized = new_dict(Device*, SpecProgram*, (HashFn) hash_device, (CmpFn) cmp_devices);
    return program;
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    destroy_arena(program->arena);
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

static bool compile_specialized_program(SpecProgram* spec) {
    CompilerConfig config = default_compiler_config();

    // TODO set subgroup size from this !
    // config.per_thread_stack_size = ...

    CHECK(run_compiler_passes(&config, &spec->arena, &spec->final_program) == CompilationNoError, return false);
    emit_spirv(&config, spec->arena, spec->final_program, &spec->spirv_size, &spec->spirv_bytes);
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

    ArenaConfig arena_config = {};
    spec_program->arena = new_arena(arena_config);
    spec_program->final_program = program->generic_program;

    CHECK(compile_specialized_program(spec_program), return NULL);
    CHECK(extract_layout(spec_program),              return NULL);
    CHECK(create_vk_pipeline(spec_program),          return NULL);
    return spec_program;
}

SpecProgram* get_specialized_program(Program* program, Device* device) {
    SpecProgram** found = find_value_dict(Device*, SpecProgram*, program->specialized, device);
    if (found)
        return *found;
    SpecProgram* spec = create_specialized_program(program, device);
    assert(spec);
    insert_dict(Device*, SpecProgram*, program->specialized, device, spec);
    return spec;
}
