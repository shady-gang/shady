#include "shady/runner/runner.h"
#include "shady/ir.h"
#include "shady/driver.h"
#include "shady/runtime/runtime.h"

#include "runner_app_common.h"

#include "log.h"
#include "arena.h"
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

static String strip_extension_suffix(String src) {
    unsigned len = strlen(src);
    char* new = calloc(sizeof(char), len + 1);
    bool copy = false;
    for (unsigned i = 0; i < len + 1; i++) {
        unsigned j = len - i;
        assert(j >= 0 && j < len + 1);
        if (src[j] == '.')
            copy = true;
        if (copy)
            new[j] = src[j];
    }
    assert(strlen(new) > 0);
    return new;
}

int main(int argc, char* argv[]) {
    shd_log_set_level(INFO);
    Args args = {
        .driver_config = shd_default_driver_config(),
        .runtime_config = shd_rn_default_config(),
    };
    cli_parse_common_app_arguments(&args.common_app_args, &argc, argv);
    shd_parse_common_args(&argc, argv);
    shd_rn_cli_parse_config(&args.runtime_config, &argc, argv);

    Runner* runtime = shd_rn_initialize(args.runtime_config);
    Device* device = shd_rn_get_device(runtime, args.common_app_args.device);
    assert(device);

    TargetConfig target_config = shd_rn_get_device_target_config(&args.driver_config.config, device);

    shd_parse_compiler_config_args(&args.driver_config.config, &argc, argv);
    shd_parse_driver_args(&args.driver_config, &argc, argv);
    shd_driver_parse_input_files(args.driver_config.input_filenames, &argc, argv);

    shd_info_print("Shady runner test starting...\n");

    Program* program;
    IrArena* arena = NULL;
    ArenaConfig aconfig = shd_default_arena_config(&target_config);
    Module* module;
    if (shd_list_count(args.driver_config.input_filenames) != 1) {
        shd_error("usage: runner_test [program]\n");
    } else {
        arena = shd_new_ir_arena(&aconfig);
        module = shd_new_module(arena, "my_module");
        int err = shd_driver_load_source_files(&args.driver_config.config, &target_config, args.driver_config.input_filenames, module);
        if (err)
            return err;
        program = shd_rn_new_program_from_module(runtime, &args.driver_config.config, module);
    }

    ShdRunnerOracleConfig oracle_config_var = { 0 };
    ShdRunnerOracleConfig* oracle_config = NULL;
    String filename = shd_read_list(String, args.driver_config.input_filenames)[0];
    String wo_suffix = strip_extension_suffix(filename);
    String filename_json = shd_fmt_string_irarena(arena, "%sjson", wo_suffix);
    free(wo_suffix);

    size_t json_size;
    char* json;
    if (shd_read_file(filename_json, &json_size, &json)) {
        shd_log_fmt(INFO, "Found matching json file\n");
        oracle_config_var = shd_runner_oracle_parse_config(arena, json);
        oracle_config = &oracle_config_var;
        free(json);
    }

    size_t num_launch_args = 0;
    LARRAY(Buffer*, buffers, oracle_config_var.num_args);
    LARRAY(void*, launch_args, oracle_config_var.num_args);

    Arena* allocator = shd_new_arena();

    if (oracle_config) {
        num_launch_args = oracle_config->num_args;
        for (size_t i = 0; i < oracle_config->num_args; i++) {
            ShdRunnerOracleArg arg_config = oracle_config->args[i];
            if (arg_config.kind == ShdRunnerOracleArg_kind_VALUE) {
                assert(arg_config.value);
                void* bytes = shd_arena_alloc(allocator, shd_rt_get_size_of_constant(arg_config.value));
                shd_rt_materialize_constant_at(bytes, arg_config.value);
                launch_args[i] = bytes;
            } else if (arg_config.kind == ShdRunnerOracleArg_kind_BUFFER) {
                buffers[i] = shd_rn_allocate_buffer_device(device, arg_config.buffer_size);
                if (arg_config.pre_pattern) {
                    void* bytes = shd_arena_alloc(allocator, shd_rt_get_size_of_constant(arg_config.pre_pattern));
                    shd_runner_oracle_prefill(bytes, arg_config.buffer_size, arg_config.pre_pattern);
                    shd_rn_copy_to_buffer(buffers[i], 0, bytes, arg_config.buffer_size);
                }
                uint64_t* bda = shd_arena_alloc(allocator, sizeof(uint64_t));
                *bda = shd_rn_get_buffer_device_pointer(buffers[i]);
                launch_args[i] = bda;
            }
        }
    }

    shd_rn_wait_completion(shd_rn_launch_kernel(program, device, args.driver_config.specialization.entry_point ? args.driver_config.specialization.entry_point : "main", 1, 1, 1, num_launch_args, launch_args, NULL));

    if (oracle_config) {
        for (size_t i = 0; i < oracle_config->num_args; i++) {
            ShdRunnerOracleArg arg_config = oracle_config->args[i];
            if (arg_config.kind == ShdRunnerOracleArg_kind_BUFFER) {
                if (arg_config.post_pattern) {
                    void* bytes = shd_arena_alloc(allocator, shd_rt_get_size_of_constant(arg_config.post_pattern));
                    shd_rn_copy_from_buffer(buffers[i], 0, bytes, arg_config.buffer_size);
                    if (!shd_runner_oracle_validate(bytes, arg_config.buffer_size, arg_config.post_pattern))
                        exit(-1);
                }

                shd_rn_destroy_buffer(buffers[i]);
            }
        }
    }

    if (oracle_config)
        shd_runner_oracle_free_config(oracle_config);

    shd_destroy_arena(allocator);

    shd_rn_shutdown(runtime);
    if (arena)
        shd_destroy_ir_arena(arena);
    else
        shd_destroy_ir_arena(shd_module_get_arena(module));
    shd_destroy_driver_config(&args.driver_config);
    return 0;
}
