#include "shader_pipeline.h"

#include "shady/cli.h"

#include "log.h"

#include <stdlib.h>

#define KiB * 1024
#define MiB * 1024 KiB

ShaderLoweringConfig shd_default_shader_target_config(void) {
    return (ShaderLoweringConfig) {
        .function_call_lowering = FCL_None,
        .per_thread_stack_size = 4 KiB,
    };
}

void shd_parse_shader_target_config_args(ShaderLoweringConfig* config, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;

        if (strcmp(argv[i], "--stack-size") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing stack size");
            config->per_thread_stack_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "--lower-function-call") == 0) {
            argv[i++] = NULL;
            char* a = argv[i];
            if (strcmp(a, "none") == 0) {
                config->function_call_lowering = FCL_None;
            } else if (strcmp(a, "software-scheduler") == 0) {
                config->function_call_lowering = FCL_SoftwareScheduler;
            } else if (strcmp(a, "rt-callable") == 0) {
                config->function_call_lowering = FCL_RT_Callables;
            } else {
                shd_log_fmt(ERROR, "Invalid '%s' argument for --lower-function-call");
                shd_log_fmt(ERROR, "Valid arguments: 'none', 'software-scheduler', 'rt-callable'");
            }
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (shd_parse_help(pargc, argv, false)) {
        shd_error_print("  --stack-size [bytes]");
        shd_error_print("  --lower-function-call <none|software-scheduler|rt-callable>");
    }

    shd_pack_remaining_args(pargc, argv);
}
