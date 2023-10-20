#include "runtime_private.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

static Program* load_program_internal(Runtime* runtime, const CompilerConfig* base_config, const char* program_src, const char* program_path) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;
    program->base_config = base_config;

    CompilerConfig config = default_compiler_config();
    ArenaConfig arena_config = default_arena_config();
    program->arena = new_ir_arena(arena_config);
    CHECK(program->arena != NULL, return false);

    program->module = new_module(program->arena, "my_module");

    if (!program_src) {
        assert(program_path);
        bool ok = read_file(program_path, NULL, &program_src);
        assert(ok);
    } else {
        assert(!program_path);
    }
    CHECK(parse_file(SrcShadyIR, strlen(program_src), program_src, program->module) == CompilationNoError, return false);
    if (program_path)
        free(program_src);

    // TODO split the compilation pipeline into generic and non-generic parts
    append_list(Program*, runtime->programs, program);
    return program;
}

Program* load_program(Runtime* runtime, const CompilerConfig* base_config, const char* program_src) {
    return load_program_internal(runtime, base_config, program_src, NULL);
}

Program* load_program_from_disk(Runtime* runtime, const CompilerConfig* base_config, const char* path) {
    return load_program_internal(runtime, base_config, NULL, path);
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    destroy_ir_arena(program->arena);
    free(program);
}
