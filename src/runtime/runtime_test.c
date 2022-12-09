#include "shady/runtime.h"
#include "shady/ir.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static const char* default_shader =
"@EntryPoint(\"compute\") @WorkgroupSize(64, 1, 1) fn main() {\n"
"    debug_printf(\"hi\");"
"    return ();\n"
"}";

enum RTTErrorCodes {
    NoError,
    MissingInputArg,
    InputFileDoesNotExist = 4,
    IncorrectLogLevel = 16,
};

typedef struct {
    CompilerConfig compiler_config;
    RuntimeConfig runtime_config;
    struct List* input_filenames;
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
        } else if (strcmp(argv[i], "--print-builtin") == 0) {
            args->compiler_config.logging.skip_builtin = false;
            i++;
        } else if (strcmp(argv[i], "--print-generated") == 0) {
            args->compiler_config.logging.skip_generated = false;
            i++;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            i++;
        } else {
            append_list(const char*, args->input_filenames, argv[i]);
        }
    }

    if (help) {
        error_print("Usage: runtime_test [source.slim]\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --print-builtin\n");
        error_print("  --print-generated\n");
        exit(help ? 0 : MissingInputArg);
    }
}

char* read_file(const char* filename);

int main(int argc, const char* argv[]) {
    set_log_level(INFO);
    Args args = {
        .input_filenames = new_list(const char*),
    };
    args.runtime_config = (RuntimeConfig) {
        .use_validation = true,
        .dump_spv = true,
    };
    process_arguments(argc, argv, &args);
    info_print("Shady runtime test starting...\n");

    Runtime* runtime = initialize_runtime(args.runtime_config);
    Device* device = get_an_device(runtime);
    assert(device);
    const char* shader = NULL;

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

    // TODO handle multiple input files properly !
    assert(num_source_files < 2);
    if (num_source_files == 1)
        shader = read_files[0];
    if (!shader)
        shader = default_shader;

    Program* program = load_program(runtime, shader);
    wait_completion(launch_kernel(program, device, 1, 1, 1, 0, NULL));
    shutdown_runtime(runtime);
}
