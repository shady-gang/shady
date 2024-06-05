#include "shady/runtime.h"
#include "shady/ir.h"
#include "shady/driver.h"

#include "runtime_app_common.h"

#include "log.h"
#include "portability.h"
#include "util.h"
#include "list.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

typedef struct {
    DriverConfig driver_config;
    RuntimeConfig runtime_config;
    CommonAppArgs common_app_args;
} Args;

static const char* default_shader =
"@EntryPoint(\"Compute\") @WorkgroupSize(SUBGROUP_SIZE, 1, 1) fn my_kernel(uniform i32 a, uniform ptr global i32 b) {\n"
"    val rb = reinterpret[u64](b);\n"
"    debug_printf(\"hi %d 0x%lx\\n\", a, rb);\n"
"    return ();\n"
"}";

int main(int argc, char* argv[]) {
    set_log_level(INFO);
    Args args = {
        .driver_config = default_driver_config(),
        .runtime_config = default_runtime_config(),
    };
    cli_parse_common_app_arguments(&args.common_app_args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_runtime_config(&args.runtime_config, &argc, argv);
    cli_parse_compiler_config_args(&args.driver_config.config, &argc, argv);
    cli_parse_input_files(args.driver_config.input_filenames, &argc, argv);

    info_print("Shady runtime test starting...\n");

    Runtime* runtime = initialize_runtime(args.runtime_config);
    Device* device = get_device(runtime, args.common_app_args.device);
    assert(device);

    Program* program;
    IrArena* arena = NULL;
    ArenaConfig aconfig = default_arena_config(&args.driver_config.config.target);
    arena = new_ir_arena(&aconfig);
    if (entries_count_list(args.driver_config.input_filenames) == 0) {
        Module* module;
        driver_load_source_file(&args.driver_config.config, SrcSlim, strlen(default_shader), default_shader, "runtime_test", &module);
        program = new_program_from_module(runtime, &args.driver_config.config, module);
    } else {
        Module* module = new_module(arena, "my_module");
        int err = driver_load_source_files(&args.driver_config, module);
        if (err)
            return err;
        program = new_program_from_module(runtime, &args.driver_config.config, module);
    }

    int32_t stuff[] = { 42, 42, 42, 42 };
    Buffer* buffer = allocate_buffer_device(device, sizeof(stuff));
    copy_to_buffer(buffer, 0, stuff, sizeof(stuff));
    copy_from_buffer(buffer, 0, stuff, sizeof(stuff));

    int32_t a0 = 42;
    uint64_t a1 = get_buffer_device_pointer(buffer);
    wait_completion(launch_kernel(program, device, args.driver_config.config.specialization.entry_point ? args.driver_config.config.specialization.entry_point : "my_kernel", 1, 1, 1, 2, (void*[]) { &a0, &a1 }, NULL));

    destroy_buffer(buffer);

    shutdown_runtime(runtime);
    if (arena)
        destroy_ir_arena(arena);
    destroy_driver_config(&args.driver_config);
    return 0;
}
