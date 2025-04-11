#include "cli.h"

#include "shady/driver.h"
#include "shady/ir.h"

#include <stdlib.h>
#include <stdbool.h>

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

void shd_pack_remaining_args(int* pargc, char** argv) {
    LARRAY(char*, nargv, *pargc);
    int nargc = 0;
    for (size_t i = 0; i < *pargc; i++) {
        if (argv[i] == NULL) continue;
        nargv[nargc++] = argv[i];
    }
    memcpy(argv,nargv, sizeof(char*) * nargc);
    *pargc = nargc;
}

void shd_parse_common_args(int* pargc, char** argv) {
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
                shd_log_set_level(DEBUGVV);
            else if (strcmp(argv[i], "debugv") == 0)
                shd_log_set_level(DEBUGV);
            else if (strcmp(argv[i], "debug") == 0)
                shd_log_set_level(DEBUG);
            else if (strcmp(argv[i], "info") == 0)
                shd_log_set_level(INFO);
            else if (strcmp(argv[i], "warn") == 0)
                shd_log_set_level(WARN);
            else if (strcmp(argv[i], "error") == 0)
                shd_log_set_level(ERROR);
            else {
                incorrect_log_level:
                shd_error_print("--log-level argument takes one of: ");
                shd_error_print("debug, info, warn,  error");
                shd_error_print("\n");
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
        shd_error_print("  --log-level debug[v[v]], info, warn, error]\n");
    }

    shd_pack_remaining_args(pargc, argv);
}

static IntSizes parse_int_size(String argv) {
    if (strcmp(argv, "8") == 0)
        return IntTy8;
    if (strcmp(argv, "16") == 0)
        return IntTy16;
    if (strcmp(argv, "32") == 0)
        return IntTy32;
    if (strcmp(argv, "64") == 0)
        return IntTy64;
    shd_error("Valid pointer sizes are 8, 16, 32 or 64.");
}

#define TARGET_CONFIG_TOGGLE_OPTIONS(F) \
F(target->memory.address_spaces[AsGeneric].allowed, native-generic-pointers) \

void shd_parse_target_args(TargetConfig* target, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {

        TARGET_CONFIG_TOGGLE_OPTIONS(PARSE_TOGGLE_OPTION)

        if (strcmp(argv[i], "--subgroup-size") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing subgroup size");
            target->subgroup_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "--word-size") == 0) {
            argv[i] = NULL;
            i++;
            target->memory.word_size = parse_int_size(argv[i]);
        } else if (strcmp(argv[i], "--pointer-size") == 0) {
            argv[i] = NULL;
            i++;
            target->memory.ptr_size = parse_int_size(argv[i]);
        } else if (strcmp(argv[i], "--no-bda") == 0) {
            target->memory.address_spaces[AsGlobal].allowed = false;
        } else if (strcmp(argv[i], "--use-native-tailcalls") == 0) {
            target->capabilities.native_tailcalls = true;
            target->memory.fn_ptr_size = IntTy64;
        } else if (strcmp(argv[i], "--use-native-fncalls") == 0) {
            target->capabilities.native_fncalls = true;
            target->memory.fn_ptr_size = IntTy64;
        } else if (strcmp(argv[i], "--use-rt-pipelines") == 0) {
            target->capabilities.native_fncalls = true;
            target->capabilities.rt_pipelines = true;
            target->memory.fn_ptr_size = IntTy64;
        } else if (strcmp(argv[i], "--force-memory-emulation") == 0) {
            target->memory.address_spaces[AsPrivate].physical = false;
            target->memory.address_spaces[AsSubgroup].physical = false;
            target->memory.address_spaces[AsShared].physical = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        shd_error_print("  --entry-point <foo>                       Selects an entry point for the program to be specialized on.\n");
        shd_error_print("  --word-size <8|16|32|64>                  Sets the word size for physical memory emulation (default=32)\n");
        shd_error_print("  --pointer-size <8|16|32|64>               Sets the pointer size for physical pointers (default=64)\n");
        shd_error_print("  --subgroup-size N                         Sets the subgroup size the program will be specialized for.\n");
        shd_error_print("  --use-native-tailcalls                    Sets the subgroup size the program will be specialized for.\n");
        shd_error_print("  --use-native-fncalls                      Sets the subgroup size the program will be specialized for.\n");
    }

    shd_pack_remaining_args(pargc, argv);
}

#define COMPILER_CONFIG_TOGGLE_OPTIONS(F) \
F(config->dynamic_scheduling, dynamic-scheduling) \
F(config->hacks.force_join_point_lifting, lift-join-points) \
F(config->optimisations.inline_everything, inline-everything) \
F(config->input_cf.restructure_with_heuristics, restructure-everything) \
F(config->input_cf.add_scope_annotations, add-scope-annotations) \
F(config->input_cf.has_scope_annotations, has-scope-annotations) \

void shd_parse_compiler_config_args(CompilerConfig* config, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;

        COMPILER_CONFIG_TOGGLE_OPTIONS(PARSE_TOGGLE_OPTION)

        if (strcmp(argv[i], "--stack-size") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing stack size");
            config->per_thread_stack_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "--printf-trace") == 0) {
            argv[i++] = NULL;
            char* s = argv[i];
            char* a = strtok(s, ",");
            while (a) {
                if (strcmp(a, "top") == 0) {
                    config->printf_trace.top_function = true;
                } else if (strcmp(a, "stack") == 0) {
                    config->printf_trace.stack_size = true;
                } else if (strcmp(a, "stack-access") == 0) {
                    config->printf_trace.stack_accesses = true;
                } else if (strcmp(a, "memory-access") == 0) {
                    config->printf_trace.memory_accesses = true;
                } else if (strcmp(a, "subgroup") == 0) {
                    config->printf_trace.subgroup_ops = true;
                } else {
                    shd_log_fmt(ERROR, "Invalid '%s' argument for --printf-trace.");
                    shd_log_fmt(ERROR, "Valid arguments: 'top', 'stack', 'stack-access', 'memory-access', 'subgroup'");
                }
                a = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--max-top-iterations") == 0) {
            argv[i] = NULL;
            i++;
            config->shader_diagnostics.max_top_iterations = atoi(argv[i]);
        } else if (strcmp(argv[i], "--inline-everything") == 0) {
            config->optimisations.inline_everything = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        shd_error_print("  --shd_print-internal                          Includes internal functions in the debug output\n");
        shd_error_print("  --shd_print-generated                         Includes generated functions in the debug output\n");
        shd_error_print("  --no-dynamic-scheduling                   Disable the built-in dynamic scheduler, restricts code to only leaf functions\n");
        shd_error_print("  --lift-join-points                        Forcefully lambda-lifts all join points. Can help with reconvergence issues.\n");
    }

    shd_pack_remaining_args(pargc, argv);
}

void shd_driver_parse_unknown_options(struct List* list, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        if (argv[i][0] != '-')
            continue;
        shd_list_append(const char*, list, argv[i]);
        argv[i] = NULL;
    }

    shd_pack_remaining_args(pargc, argv);
}

void shd_driver_parse_input_files(struct List* list, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        shd_list_append(const char*, list, argv[i]);
        argv[i] = NULL;
    }

    shd_pack_remaining_args(pargc, argv);
    assert(*pargc == 1);
}

DriverConfig shd_default_driver_config(void) {
    return (DriverConfig) {
        .config = shd_default_compiler_config(),
        .target_type = TgtAuto,
        .input_filenames = shd_new_list(const char*),
        .output_filename = NULL,
        .cfg_output_filename = NULL,
        .shd_output_filename = NULL,
        .backend_config.c = shd_default_c_backend_config(),
        .backend_config.spirv = shd_default_spirv_backend_config(),
    };
}

void shd_destroy_driver_config(DriverConfig* config) {
    shd_destroy_list(config->input_filenames);
}

void shd_parse_driver_args(DriverConfig* args, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                shd_error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            args->output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                shd_error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->cfg_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-loop-tree") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                shd_error_print("--dump-loop-tree must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->loop_tree_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                shd_error_print("--dump-ir must be followed with a filename");
                exit(MissingDumpIrArg);
            }
            args->shd_output_filename = argv[i];
        } else if (strcmp(argv[i], "--entry-point") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing entry point name");
            args->specialization.entry_point = argv[i];
        } else if (strcmp(argv[i], "--execution-model") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing execution model name");
            ExecutionModel em = EmNone;
#define EM(n) if (strcmp(argv[i], #n) == 0) em = Em##n;
            EXECUTION_MODELS(EM)
#undef EM
            if (em == EmNone)
                shd_error("Unknown execution model: %s", argv[i]);
            switch (em) {
                case EmFragment:
                case EmVertex:
                    args->config.dynamic_scheduling = false;
                break;
                default: break;
            }
            args->specialization.execution_model = em;
        } else if (strcmp(argv[i], "--glsl-version") == 0) {
            argv[i] = NULL;
            i++;
            args->backend_config.c.glsl_version = strtol(argv[i], NULL, 10);
        } else if (strcmp(argv[i], "--target") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                goto invalid_target;
            else if (strcmp(argv[i], "c") == 0)
                args->target_type = TgtC;
            else if (strcmp(argv[i], "spirv") == 0)
                args->target_type = TgtSPV;
            else if (strcmp(argv[i], "glsl") == 0)
                args->target_type = TgtGLSL;
            else if (strcmp(argv[i], "ispc") == 0)
                args->target_type = TgtISPC;
            else if (strcmp(argv[i], "cuda") == 0)
                args->target_type = TgtCUDA;
            else
                goto invalid_target;
            argv[i] = NULL;
            continue;
            invalid_target:
            shd_error_print("--target must be followed with a valid target (see help for list of targets)");
            exit(InvalidTarget);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
#define EM(name) #name", "
            shd_error_print("  --execution-model <em>                   Selects an entry point for the program to be specialized on.\nPossible values: " EXECUTION_MODELS(EM) "\n");
#undef EM
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
