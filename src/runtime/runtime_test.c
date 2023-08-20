#include "shady/runtime.h"
#include "shady/ir.h"
#include "shady/cli.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static const char* default_shader =
"@EntryPoint(\"compute\") @WorkgroupSize(SUBGROUP_SIZE, 1, 1) fn main(uniform i32 a, uniform ptr global i32 b) {\n"
"    debug_printf(\"hi %d %p\", a, b);"
"    return ();\n"
"}";

typedef struct {
    CompilerConfig compiler_config;
    RuntimeConfig runtime_config;
    struct List* input_filenames;
    size_t device;
} Args;

static void parse_runtime_arguments(int* pargc, char** argv, Args* args) {
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
        error_print("  --print-builtin\n");
        error_print("  --print-generated\n");
        error_print("  --device n\n");
        exit(0);
    }
}

int main(int argc, char* argv[]) {
    set_log_level(INFO);
    Args args = {
        .input_filenames = new_list(const char*),
        .compiler_config = default_compiler_config(),
    };
    args.runtime_config = (RuntimeConfig) {
        .use_validation = true,
        .dump_spv = true,
    };
    parse_runtime_arguments(&argc, argv, &args);
    parse_common_args(&argc, argv);
    parse_compiler_config_args(&args.compiler_config, &argc, argv);
    parse_input_files(args.input_filenames, &argc, argv);

    info_print("Shady runtime test starting...\n");

    Runtime* runtime = initialize_runtime(args.runtime_config);
    Device* device = get_device(runtime, args.device);
    assert(device);
    const char* shader = NULL;

    // Read the files
    size_t num_source_files = entries_count_list(args.input_filenames);
    LARRAY(const char*, read_files, num_source_files);
    for (size_t i = 0; i < num_source_files; i++) {
        unsigned char* input_file_contents;

        bool ok = read_file(read_list(const char*, args.input_filenames)[i], NULL, &input_file_contents);
        assert(ok);
        if (input_file_contents == NULL) {
            error_print("file does not exist\n");
            exit(InputFileDoesNotExist);
        }
        read_files[i] = (char*)input_file_contents;
    }
    destroy_list(args.input_filenames);

    // TODO handle multiple input files properly !
    assert(num_source_files < 2);
    if (num_source_files == 1)
        shader = read_files[0];
    if (!shader)
        shader = default_shader;

    int32_t stuff[] = { 42, 42, 42, 42 };
    Buffer* buffer = allocate_buffer_device(device, sizeof(stuff));
    copy_to_buffer(buffer, 0, stuff, sizeof(stuff));
    copy_from_buffer(buffer, 0, stuff, sizeof(stuff));

    Program* program = load_program(runtime, &args.compiler_config, shader);

    int32_t a0 = 42;
    uint64_t a1 = get_buffer_device_pointer(buffer);
    wait_completion(launch_kernel(program, device, "main", 1, 1, 1, 2, (void*[]) { &a0, &a1 }));

    destroy_buffer(buffer);

    shutdown_runtime(runtime);
}
