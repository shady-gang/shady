#include "shady/runtime.h"
#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "portability.h"
#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static const char* default_shader =
"@EntryPoint(\"Compute\") @WorkgroupSize(SUBGROUP_SIZE, 1, 1) fn main(uniform i32 a, uniform ptr global i32 b) {\n"
"    val rb = reinterpret[u64](b);\n"
"    debug_printf(\"hi %d 0x%lx\\n\", a, rb);\n"
"    return ();\n"
"}";

typedef struct {
    DriverConfig driver_config;
    RuntimeConfig runtime_config;
    size_t device;
} Args;

static void parse_runtime_arguments(int* pargc, char** argv, Args* args) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else if (strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) {
            argv[i] = NULL;
            i++;
            args->device = strtol(argv[i], NULL, 10);
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        error_print("Usage: runtime_test [source.slim]\n");
        error_print("Available arguments: \n");
        error_print("  --log-level debug[v[v]], info, warn, error]\n");
        error_print("  --print-builtin\n");
        error_print("  --print-generated\n");
        error_print("  --device n\n");
        exit(0);
    }
}

int main(int argc, char* argv[]) {
    set_log_level(INFO);
    Args args = {
        .driver_config = default_driver_config(),
    };
    args.runtime_config = (RuntimeConfig) {
        .use_validation = true,
        .dump_spv = true,
    };
    parse_runtime_arguments(&argc, argv, &args);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.driver_config.config, &argc, argv);
    cli_parse_input_files(args.driver_config.input_filenames, &argc, argv);

    info_print("Shady runtime test starting...\n");

    Runtime* runtime = initialize_runtime(args.runtime_config);
    Device* device = get_device(runtime, args.device);
    assert(device);

    IrArena* arena = new_ir_arena(default_arena_config());
    Module* module = new_module(arena, "my_module");

    int err = driver_load_source_files(&args.driver_config, module);
    if (err)
        return err;
    Program* program = new_program_from_module(runtime, &args.driver_config.config, module);

    int32_t stuff[] = { 42, 42, 42, 42 };
    Buffer* buffer = allocate_buffer_device(device, sizeof(stuff));
    copy_to_buffer(buffer, 0, stuff, sizeof(stuff));
    copy_from_buffer(buffer, 0, stuff, sizeof(stuff));

    int32_t a0 = 42;
    uint64_t a1 = get_buffer_device_pointer(buffer);
    wait_completion(launch_kernel(program, device, "main", 1, 1, 1, 2, (void*[]) { &a0, &a1 }));

    destroy_buffer(buffer);

    shutdown_runtime(runtime);
    destroy_ir_arena(arena);
    destroy_driver_config(&args.driver_config);
    return 0;
}
