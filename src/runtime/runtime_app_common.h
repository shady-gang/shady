#ifndef SHADY_RUNTIME_CLI
#define SHADY_RUNTIME_CLI

#include "shady/driver.h"
#include "runtime_private.h"

#include "log.h"

#include <string.h>
#include <stdlib.h>

typedef struct {
    size_t device;
} CommonAppArgs;

static void cli_parse_common_app_arguments(CommonAppArgs* args, int* pargc, char** argv) {
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
            if (i >= argc) {
                error_print("Missing device number for --device\n");
                exit(1);
            }
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
        error_print("  --shd_print-builtin\n");
        error_print("  --shd_print-generated\n");
        error_print("  --device n\n");
        exit(0);
    }

    cli_pack_remaining_args(pargc, argv);
}

#endif
