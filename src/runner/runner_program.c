#include "runner_private.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

Program* shd_rt_new_program_from_module(Runtime* runtime, const CompilerConfig* base_config, Module* mod) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;
    program->base_config = base_config;
    program->arena = NULL;
    program->module = mod;

    // TODO split the compilation pipeline into generic and non-generic parts
    shd_list_append(Program*, runtime->programs, program);
    return program;
}

Program* shd_rt_load_program(Runtime* runtime, const CompilerConfig* base_config, const char* program_src) {
    Module* module;

    int err = shd_driver_load_source_file(base_config, SrcShadyIR, strlen(program_src), program_src, "my_module",
                                          &module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = shd_rt_new_program_from_module(runtime, base_config, module);
    program->arena = shd_module_get_arena(module);
    return program;
}

Program* shd_rt_load_program_from_disk(Runtime* runtime, const CompilerConfig* base_config, const char* path) {
    Module* module;

    int err = shd_driver_load_source_file_from_filename(base_config, path, "my_module", &module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = shd_rt_new_program_from_module(runtime, base_config, module);
    program->arena = shd_module_get_arena(module);
    return program;
}

void shd_rt_unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    if (program->arena) // if the program owns an arena
        shd_destroy_ir_arena(program->arena);
    free(program);
}
