#include "shady/cli.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "log.h"
#include "portability.h"
#include "list.h"

bool string_ends_with(const char* string, const char* suffix) {
    size_t len = strlen(string);
    size_t slen = strlen(suffix);
    if (len < slen)
        return false;
    for (size_t i = 0; i < slen; i++) {
        if (string[len - 1 - i] != suffix[slen - 1 - i])
            return false;
    }
    return true;
}

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

void pack_remaining_args(int* pargc, char** argv) {
    LARRAY(char*, nargv, *pargc);
    int nargc = 0;
    for (size_t i = 0; i < *pargc; i++) {
        if (argv[i] == NULL) continue;
        nargv[nargc++] = argv[i];
    }
    memcpy(argv,nargv, sizeof(char*) * nargc);
    *pargc = nargc;
}

void parse_common_args(int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
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

    pack_remaining_args(pargc, argv);
}

void parse_compiler_config_args(CompilerConfig* config, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-dynamic-scheduling") == 0) {
            config->dynamic_scheduling = false;
        } else if (strcmp(argv[i], "--simt2d") == 0) {
            config->lower.simt_to_explicit_simd = true;
        } else if (strcmp(argv[i], "--print-builtin") == 0) {
            config->logging.skip_builtin = false;
        } else if (strcmp(argv[i], "--print-generated") == 0) {
            config->logging.skip_generated = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        error_print("  --print-builtin                           Includes builtin-in functions in the debug output\n");
        error_print("  --print-generated                         Includes generated functions in the debug output\n");
        error_print("  --no-dynamic-scheduling                   Disable the built-in dynamic scheduler, restricts code to only leaf functions\n");
        error_print("  --simt2d                                  Emits SIMD code instead of SIMT, only effective with the C backend.\n");
    }

    pack_remaining_args(pargc, argv);
}

void parse_input_files(struct List* list, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        append_list(const char*, list, argv[i]);
        argv[i] = NULL;
    }

    pack_remaining_args(pargc, argv);
    assert(*pargc == 1);
}
