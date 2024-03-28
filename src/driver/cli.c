#include "shady/driver.h"
#include "shady/ir.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

CodegenTarget guess_target(const char* filename) {
    if (string_ends_with(filename, ".c"))
        return TgtC;
    else if (string_ends_with(filename, "glsl"))
        return TgtGLSL;
    else if (string_ends_with(filename, "spirv") || string_ends_with(filename, "spv"))
        return TgtSPV;
    else if (string_ends_with(filename, "ispc"))
        return TgtISPC;
    error_print("No target has been specified, and output filename '%s' did not allow guessing the right one\n");
    exit(InvalidTarget);
}

void cli_pack_remaining_args(int* pargc, char** argv) {
    LARRAY(char*, nargv, *pargc);
    int nargc = 0;
    for (size_t i = 0; i < *pargc; i++) {
        if (argv[i] == NULL) continue;
        nargv[nargc++] = argv[i];
    }
    memcpy(argv,nargv, sizeof(char*) * nargc);
    *pargc = nargc;
}

void cli_parse_common_args(int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        if (strcmp(argv[i], "--log-level") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                goto incorrect_log_level;
            if (strcmp(argv[i], "debugvv") == 0)
                set_log_level(DEBUGVV);
            else if (strcmp(argv[i], "debugv") == 0)
                set_log_level(DEBUGV);
            else if (strcmp(argv[i], "debug") == 0)
                set_log_level(DEBUG);
            else if (strcmp(argv[i], "info") == 0)
                set_log_level(INFO);
            else if (strcmp(argv[i], "warn") == 0)
                set_log_level(WARN);
            else if (strcmp(argv[i], "error") == 0)
                set_log_level(ERROR);
            else {
                incorrect_log_level:
                error_print("--log-level argument takes one of: ");
                error_print("debug, info, warn,  error");
                error_print("\n");
                exit(IncorrectLogLevel);
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        error_print("  --log-level debug[v[v]], info, warn, error]\n");
    }

    cli_pack_remaining_args(pargc, argv);
}

#define PARSE_TOGGLE_OPTION(f, name) \
if (strcmp(argv[i], "--no-"#name) == 0) { \
    config->f = false; argv[i] = NULL; continue; \
} else if (strcmp(argv[i], "--"#name) == 0) { \
    config->f = true; argv[i] = NULL; continue; \
}

#define TOGGLE_OPTIONS(F) \
F(lower.emulate_physical_memory, emulate-physical-memory) \
F(dynamic_scheduling, dynamic-scheduling) \
F(hacks.force_join_point_lifting, lift-join-points) \
F(hacks.assume_no_physical_global_ptrs, assume-no-physical-global-ptrs) \
F(logging.print_internal, print-internal) \
F(logging.print_generated, print-builtin) \
F(logging.print_generated, print-generated) \
F(lower.simt_to_explicit_simd, lower-simt-to-simd) \

void cli_parse_compiler_config_args(CompilerConfig* config, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;

        TOGGLE_OPTIONS(PARSE_TOGGLE_OPTION)

        if (strcmp(argv[i], "--entry-point") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                error("Missing entry point name");
            config->specialization.entry_point = argv[i];
        } else if (strcmp(argv[i], "--subgroup-size") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                error("Missing subgroup size name");
            config->specialization.subgroup_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "--execution-model") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                error("Missing execution model name");
            ExecutionModel em = EmNone;
#define EM(n, _) if (strcmp(argv[i], #n) == 0) em = Em##n;
            EXECUTION_MODELS(EM)
#undef EM
            if (em == EmNone)
                error("Unknown execution model: %s", argv[i]);
            switch (em) {
                case EmFragment:
                case EmVertex:
                    config->dynamic_scheduling = false;
                    break;
                default: break;
            }
            config->specialization.execution_model = em;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        error_print("  --print-internal                          Includes internal functions in the debug output\n");
        error_print("  --print-generated                         Includes generated functions in the debug output\n");
        error_print("  --no-dynamic-scheduling                   Disable the built-in dynamic scheduler, restricts code to only leaf functions\n");
        error_print("  --simt2d                                  Emits SIMD code instead of SIMT, only effective with the C backend.\n");
        error_print("  --entry-point <foo>                       Selects an entry point for the program to be specialized on.\n");
#define EM(name, _) #name", "
        error_print("  --execution-model <em>                   Selects an entry point for the program to be specialized on.\nPossible values: " EXECUTION_MODELS(EM));
#undef EM
        error_print("  --subgroup-size N                         Sets the subgroup size the program will be specialized for.\n");
        error_print("  --lift-join-points                        Forcefully lambda-lifts all join points. Can help with reconvergence issues.\n");
    }

    cli_pack_remaining_args(pargc, argv);
}

void cli_parse_input_files(struct List* list, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        append_list(const char*, list, argv[i]);
        argv[i] = NULL;
    }

    cli_pack_remaining_args(pargc, argv);
    assert(*pargc == 1);
}

DriverConfig default_driver_config() {
    return (DriverConfig) {
        .config = default_compiler_config(),
        .target = TgtAuto,
        .input_filenames = new_list(const char*),
        .output_filename = NULL,
        .cfg_output_filename = NULL,
        .shd_output_filename = NULL,
    };
}

void destroy_driver_config(DriverConfig* config) {
    destroy_list(config->input_filenames);
}

void cli_parse_driver_arguments(DriverConfig* args, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            args->output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->cfg_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-loop-tree") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-loop-tree must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->loop_tree_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-ir must be followed with a filename");
                exit(MissingDumpIrArg);
            }
            args->shd_output_filename = argv[i];
        } else if (strcmp(argv[i], "--target") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                goto invalid_target;
            else if (strcmp(argv[i], "c") == 0)
                args->target = TgtC;
            else if (strcmp(argv[i], "spirv") == 0)
                args->target = TgtSPV;
            else if (strcmp(argv[i], "glsl") == 0)
                args->target = TgtGLSL;
            else if (strcmp(argv[i], "ispc") == 0)
                args->target = TgtISPC;
            else
                goto invalid_target;
            argv[i] = NULL;
            continue;
            invalid_target:
            error_print("--target must be followed with a valid target (see help for list of targets)");
            exit(InvalidTarget);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        // error_print("Usage: slim source.slim\n");
        // error_print("Available arguments: \n");
        error_print("  --target <c, glsl, ispc, spirv>           \n");
        error_print("  --output <filename>, -o <filename>        \n");
        error_print("  --dump-cfg <filename>                     Dumps the control flow graph of the final IR\n");
        error_print("  --dump-loop-tree <filename>\n");
        error_print("  --dump-ir <filename>                      Dumps the final IR\n");
    }

    cli_pack_remaining_args(pargc, argv);
}