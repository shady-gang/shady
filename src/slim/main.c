#include "shady/ir.h"

#include "../log.h"
#include "../passes/passes.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

const char* input_filename = NULL;
const char* output_filename = "out.spv";

const char* cfg_output = NULL;

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MoreThanOneFilename,
    MissingDumpCfgArg
};

char* read_file(const char* filename);

static void process_arguments(int argc, const char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log-level") == 0) {
            i++;
            if (i == argc)
                goto incorrect_log_level;
            if (strcmp(argv[i], "debug") == 0)
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
        } else if (strcmp(argv[i], "--output") == 0) {
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            cfg_output = argv[i];
        } else {
            // assume it is the filename
            if (input_filename) {
                error_print("We only supporting compiling from one file at the moment");
                exit(MoreThanOneFilename);
            }
            input_filename = argv[i];
        }
    }

    if (input_filename == NULL) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --output output_filename\n");
        error_print("  --dump-cfg\n");
        exit(MissingInputArg);
    }
}

int main(int argc, const char** argv) {
    IrArena* arena = new_arena((IrConfig) {
        .check_types = false
    });

    CompilerConfig config = default_compiler_config();

    process_arguments(argc, argv);

    info_print("compiling %s\n", input_filename);
    const Node* program;

    char* contents = read_file(input_filename);
    if ((void*)contents == NULL) {
        error_print("file does not exist\n");
        exit(InputFileDoesNotExist);
    } else {
        info_print("Parsing: \n%s\n", contents);
        program = parse(contents, arena);
    }

    free(contents);

    info_print("Parsed program successfully: \n");
    print_node(program);

    CompilationResult result = run_compiler_passes(&config, &arena, &program);
    if (result != CompilationNoError) {
        error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }

    if (cfg_output) {
        FILE* cfg_output_f = fopen(cfg_output, "wb");
        assert(cfg_output_f);
        dump_cfg(cfg_output_f, program);
        fclose(cfg_output_f);
    }

    info_print("Emitting final result ... \n");
    FILE *output = fopen(output_filename, "wb");
    emit_spirv(&config, arena, program, output);
    fclose(output);
    info_print("Done\n");

    destroy_arena(arena);
    return NoError;
}
