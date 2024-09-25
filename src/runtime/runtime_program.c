#include "runtime_private.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../shady/transform/memory_layout.h"

Program* new_program_from_module(Runtime* runtime, const CompilerConfig* base_config, Module* mod) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;
    program->base_config = base_config;
    program->arena = NULL;
    program->module = mod;

    // TODO split the compilation pipeline into generic and non-generic parts
    shd_list_append(Program*, runtime->programs, program);
    return program;
}

Program* load_program(Runtime* runtime, const CompilerConfig* base_config, const char* program_src) {
    Module* module;

    int err = driver_load_source_file(base_config, SrcShadyIR, strlen(program_src), program_src, "my_module", &module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = new_program_from_module(runtime, base_config, module);
    program->arena = get_module_arena(module);
    return program;
}

Program* load_program_from_disk(Runtime* runtime, const CompilerConfig* base_config, const char* path) {
    Module* module;

    int err = driver_load_source_file_from_filename(base_config, path, "my_module", &module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = new_program_from_module(runtime, base_config, module);
    program->arena = get_module_arena(module);
    return program;
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    if (program->arena) // if the program owns an arena
        destroy_ir_arena(program->arena);
    free(program);
}
