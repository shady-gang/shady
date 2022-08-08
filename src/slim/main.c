#include "shady/ir.h"

#include "list.h"

#include "log.h"
#include "portability.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

char* read_file(const char* filename);

const char* cfg_output = NULL;

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MissingDumpCfgArg
};

static void process_arguments(int argc, const char** argv, struct List* input_filenames, const char** output_filename) {
    bool help = false;
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
            *output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            cfg_output = argv[i];
        }  else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            i++;
        } else {
            append_list(const char*, input_filenames, argv[i]);
        }
    }

    if (entries_count_list(input_filenames) == 0 || help) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --output output_filename\n");
        error_print("  --dump-cfg\n");
        exit(help ? 0 : MissingInputArg);
    }
}

int main(int argc, const char** argv) {
    platform_specific_terminal_init_extras();

    IrArena* arena = new_arena((ArenaConfig) {
        .check_types = false
    });

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;

    struct List* input_files = new_list(const char*);
    const char* output_filename = "out.spv";
    process_arguments(argc, argv, input_files, &output_filename);

    // Read the files
    size_t num_source_files = entries_count_list(input_files);
    LARRAY(const char*, read_files, num_source_files);
    for (size_t i = 0; i < num_source_files; i++) {
        const char* input_file_contents = read_file(read_list(const char*, input_files)[i]);
        if ((void*)input_file_contents == NULL) {
            error_print("file does not exist\n");
            exit(InputFileDoesNotExist);
        }
        read_files[i] = input_file_contents;
    }
    destroy_list(input_files);

    // Parse the lot
    const Node* program = NULL;
    CompilationResult parse_result = parse_files(&config, num_source_files, read_files, arena, &program);
    assert(parse_result == CompilationNoError);

    // Free the read files
    for (size_t i = 0; i < num_source_files; i++)
        free((void*) read_files[i]);

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
    FILE* output_file = fopen(output_filename, "wb");
    size_t output_size;
    char* output_buffer;
    emit_spirv(&config, arena, program, &output_size, &output_buffer);
    fwrite(output_buffer, output_size, 1, output_file);
    free((void*) output_buffer);
    fclose(output_file);
    info_print("Done\n");

    destroy_arena(arena);
    return NoError;
}
