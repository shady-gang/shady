#include "shady/runner/runner.h"
#include "shady/ir.h"
#include "shady/driver.h"

#include "runner_app_common.h"

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
    RunnerConfig runtime_config;
    CommonAppArgs common_app_args;
} Args;

static const char* default_shader =
"@EntryPoint(\"Compute\") @Exported @WorkgroupSize(SUBGROUP_SIZE, 1, 1) fn my_kernel(i32 a, ptr global i32 b) {\n"
"    val rb = bitcast[u64](b);\n"
"    debug_printf(\"hi %d 0x%lx\\n\", a, rb);\n"
"    return ();\n"
"}";

int main(int argc, char* argv[]) {
    shd_log_set_level(INFO);
    Args args = {
        .driver_config = shd_default_driver_config(),
        .runtime_config = shd_rn_default_config(),
    };
    cli_parse_common_app_arguments(&args.common_app_args, &argc, argv);
    shd_parse_common_args(&argc, argv);
    shd_rn_cli_parse_config(&args.runtime_config, &argc, argv);
    shd_parse_compiler_config_args(&args.driver_config.config, &argc, argv);
    shd_driver_parse_input_files(args.driver_config.input_filenames, &argc, argv);

    shd_info_print("Shady runner test starting...\n");

    Runner* runtime = shd_rn_initialize(args.runtime_config);
    Device* device = shd_rn_get_device(runtime, args.common_app_args.device);
    assert(device);

    Program* program;
    IrArena* arena = NULL;
    ArenaConfig aconfig = shd_default_arena_config(&args.driver_config.config.target);
    Module* module;
    if (shd_list_count(args.driver_config.input_filenames) == 0) {
        shd_driver_load_source_file(&args.driver_config.config, SrcSlim, strlen(default_shader), default_shader,
                                    "runtime_test", &module);
        program = shd_rn_new_program_from_module(runtime, &args.driver_config.config, module);
    } else {
        arena = shd_new_ir_arena(&aconfig);
        module = shd_new_module(arena, "my_module");
        int err = shd_driver_load_source_files(&args.driver_config, module);
        if (err)
            return err;
        program = shd_rn_new_program_from_module(runtime, &args.driver_config.config, module);
    }

    int32_t stuff[] = { 42, 42, 42, 42 };
    Buffer* buffer = shd_rn_allocate_buffer_device(device, sizeof(stuff));
    shd_rn_copy_to_buffer(buffer, 0, stuff, sizeof(stuff));
    shd_rn_copy_from_buffer(buffer, 0, stuff, sizeof(stuff));

    int32_t a0 = 42;
    uint64_t a1 = shd_rn_get_buffer_device_pointer(buffer);
    shd_rn_wait_completion(shd_rn_launch_kernel(program, device, args.driver_config.config.specialization.entry_point ? args.driver_config.config.specialization.entry_point : "my_kernel", 1, 1, 1, 2, (void* []) { &a0, &a1 }, NULL));

    shd_rn_destroy_buffer(buffer);

    shd_rn_shutdown(runtime);
    if (arena)
        shd_destroy_ir_arena(arena);
    else
        shd_destroy_ir_arena(shd_module_get_arena(module));
    shd_destroy_driver_config(&args.driver_config);
    return 0;
}
