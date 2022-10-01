#include "shady/ir.h"

#include "list.h"

#include "log.h"
#include "portability.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

char* read_file(const char* filename);

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MissingDumpCfgArg,
    MissingDumpIrArg,
};

typedef struct {
     struct List* input_filenames;
     const char* spv_output_filename;
     const char* shd_output_filename;
     const char* cfg_output_filename;
} Args;

static void process_arguments(int argc, const char** argv, Args* args) {
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
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            args->spv_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->cfg_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-ir must be followed with a filename");
                exit(MissingDumpIrArg);
            }
            args->shd_output_filename = argv[i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            i++;
        } else {
            append_list(const char*, args->input_filenames, argv[i]);
        }
    }

    if (entries_count_list(args->input_filenames) == 0 || help) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --output <filename>, -o <filename>\n");
        error_print("  --dump-cfg <filename>\n");
        error_print("  --dump-ir <filename>\n");
        exit(help ? 0 : MissingInputArg);
    }
}

int main(int argc, const char** argv) {
    platform_specific_terminal_init_extras();

    IrArena* arena = new_ir_arena((ArenaConfig) {
        .check_types = false
    });

    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;

    Args args = {
        .input_filenames = new_list(const char*),
        .spv_output_filename = "out.spv",
        .cfg_output_filename = NULL,
        .shd_output_filename = NULL,
    };
    process_arguments(argc, argv, &args);

    // Read the files
    size_t num_source_files = entries_count_list(args.input_filenames);
    LARRAY(const char*, read_files, num_source_files);
    for (size_t i = 0; i < num_source_files; i++) {
        const char* input_file_contents = read_file(read_list(const char*, args.input_filenames)[i]);
        if ((void*)input_file_contents == NULL) {
            error_print("file does not exist\n");
            exit(InputFileDoesNotExist);
        }
        read_files[i] = input_file_contents;
    }
    destroy_list(args.input_filenames);

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
    info_print("Ran all passes successfully\n");

    if (args.cfg_output_filename) {
        FILE* f = fopen(args.cfg_output_filename, "wb");
        assert(f);
        dump_cfg(f, program);
        fclose(f);
        info_print("CFG dumped\n");
    }

    if (args.shd_output_filename) {
        FILE* f = fopen(args.shd_output_filename, "wb");
        assert(f);
        size_t output_size;
        char* output_buffer;
        print_node_into_str(program, &output_buffer, &output_size);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
        info_print("IR dumped\n");
    }

    if (args.spv_output_filename) {
        FILE* f = fopen(args.spv_output_filename, "wb");
        size_t output_size;
        char* output_buffer;
        emit_spirv(&config, arena, program, &output_size, &output_buffer);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
    }
    info_print("Done\n");

    destroy_ir_arena(arena);
    return NoError;
}
