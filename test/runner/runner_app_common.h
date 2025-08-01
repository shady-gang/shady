#ifndef SHADY_RUNTIME_CLI
#define SHADY_RUNTIME_CLI

#include "shady/driver.h"
#include "shady/runner/runner.h"

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
                shd_error_print("Missing device number for --device\n");
                exit(1);
            }
            args->device = strtol(argv[i], NULL, 10);
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        shd_error_print("Usage: runtime_test [source.slim]\n");
        shd_error_print("Available arguments: \n");
        shd_error_print("  --device n\n");
        exit(0);
    }

    shd_pack_remaining_args(pargc, argv);
}

typedef struct {
    enum { ShdRunnerOracleArg_kind_VALUE, ShdRunnerOracleArg_kind_BUFFER } kind;
    uint32_t buffer_size;
    const Type* type;
    const Node* value;
    const Node* pre_pattern;
    const Node* post_pattern;
} ShdRunnerOracleArg;

typedef struct {
    uint32_t dispatch_size[3];
    uint32_t num_args;
    ShdRunnerOracleArg* args;
} ShdRunnerOracleConfig;

ShdRunnerOracleConfig shd_runner_oracle_parse_config(IrArena* a, String json);
void shd_runner_oracle_free_config(ShdRunnerOracleConfig*);

void shd_runner_oracle_prefill(void* dst, size_t size, const Node* pattern);
bool shd_runner_oracle_validate(void* dst, size_t size, const Node* pattern);

#endif
