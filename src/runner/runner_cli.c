#include "runner_private.h"
#include "../driver/cli.h"

#include "log.h"

RuntimeConfig shd_rt_default_config() {
    return (RuntimeConfig) {
#ifndef NDEBUG
        .dump_spv = true,
        .use_validation = true,
#else
        0
#endif
    };
}

#define DRIVER_CONFIG_OPTIONS(F) \
F(config->use_validation, api-validation) \
F(config->dump_spv, dump-spv) \

void shd_rt_cli_parse_runtime_config(RuntimeConfig* config, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {

        DRIVER_CONFIG_OPTIONS(PARSE_TOGGLE_OPTION)

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        // shd_error_print("Usage: slim source.slim\n");
        // shd_error_print("Available arguments: \n");
        shd_error_print("  --target <c, glsl, ispc, spirv>           \n");
        shd_error_print("  --output <filename>, -o <filename>        \n");
        shd_error_print("  --dump-cfg <filename>                     Dumps the control flow graph of the final IR\n");
        shd_error_print("  --dump-loop-tree <filename>\n");
        shd_error_print("  --dump-ir <filename>                      Dumps the final IR\n");
    }

    shd_pack_remaining_args(pargc, argv);
}