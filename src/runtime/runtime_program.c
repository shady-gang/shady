#include "runtime_private.h"
#include "shady/ir.h"

#include "log.h"
#include "list.h"
#include <stdlib.h>
#include <assert.h>

static Program* load_program_internal(Runtime* runtime, const char* program_src, const char* program_path) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;
    ArenaConfig arena_config = default_arena_config();
    program->arena = new_ir_arena(arena_config);
    CHECK(program->arena != NULL, return false);

    program->module = new_module(program->arena, "my_module");

    if (program_src) {
        CHECK(parse_files(&config, 1, NULL, (const char* []) {program_src}, program->module) == CompilationNoError, return false);
    } else if (program_path) {
        CHECK(parse_files(&config, 1, (const char* []) { program_path}, NULL, program->module) == CompilationNoError, return false);
    } else {
        assert(false);
    }

    // TODO split the compilation pipeline into generic and non-generic parts
    append_list(Program*, runtime->programs, program);
    return program;
}

Program* load_program(Runtime* runtime, const char* program_src) {
    return load_program_internal(runtime, program_src, NULL);
}

Program* load_program_from_disk(Runtime* runtime, const char* path) {
    return load_program_internal(runtime, NULL, path);
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    destroy_ir_arena(program->arena);
    free(program);
}
