#include "runner_private.h"
#include "shady/driver.h"
#include "shady/ir/module.h"
#include "shady/ir/arena.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

Program* shd_rn_new_program_from_module(Runner* runtime, const CompilerConfig* base_config, Module* mod) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;
    program->base_config = base_config;
    program->arena = NULL;
    program->module = mod;

    // TODO split the compilation pipeline into generic and non-generic parts
    shd_list_append(Program*, runtime->programs, program);
    return program;
}

Program* shd_rt_load_program(Runner* runtime, const CompilerConfig* base_config, const char* program_src) {
    Module* module;
    TargetConfig target_config = shd_default_target_config();

    int err = shd_driver_load_source_file(base_config, &target_config, SrcShadyIR, strlen(program_src), program_src, "my_module", &module);
    if (err != ShdNoError) {
        return NULL;
    }

    Program* program = shd_rn_new_program_from_module(runtime, base_config, module);
    program->arena = shd_module_get_arena(module);
    return program;
}

Program* shd_rt_load_program_from_disk(Runner* runtime, const CompilerConfig* base_config, const char* path) {
    Module* module;
    TargetConfig target_config = shd_default_target_config();

    int err = shd_driver_load_source_file_from_filename(base_config, &target_config, path, "my_module", &module);
    if (err != ShdNoError) {
        return NULL;
    }

    Program* program = shd_rn_new_program_from_module(runtime, base_config, module);
    program->arena = shd_module_get_arena(module);
    return program;
}

void shd_rn_unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    if (program->arena) // if the program owns an arena
        shd_destroy_ir_arena(program->arena);
    free(program);
}
