#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "growy.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

typedef struct {
    char* tmp_filename;
    bool delete_tmp_file;
    char* include_path;
    bool only_run_clang;
} VccOptions;

static void cli_parse_vcc_args(VccOptions* options, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        else if (strcmp(argv[i], "--vcc-keep-tmp-file") == 0) {
            argv[i] = NULL;
            options->delete_tmp_file = false;
            options->tmp_filename = "vcc_tmp.ll";
            continue;
        } else if (strcmp(argv[i], "--vcc-include-path") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                error("Missing subgroup size name");
            options->include_path = argv[i];
            continue;
        } else if (strcmp(argv[i], "--only-run-clang") == 0) {
            argv[i] = NULL;
            options->only_run_clang = true;
            continue;
        }
    }

    cli_pack_remaining_args(pargc, argv);
}

uint32_t hash_murmur(const void* data, size_t size);

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    VccOptions vcc_options = {
        .tmp_filename = NULL,
        .delete_tmp_file = true
    };
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_vcc_args(&vcc_options, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    if (entries_count_list(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    ArenaConfig aconfig = default_arena_config();
    IrArena* arena = new_ir_arena(aconfig);
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    int clang_retval = system("clang --version");
    if (clang_retval != 0)
    error("clang not present in path or otherwise broken (retval=%d)", clang_retval);

    size_t num_source_files = entries_count_list(args.input_filenames);

    Growy* g = new_growy();
    growy_append_string(g, "clang");
    char* self_path = get_executable_location();
    char* working_dir = strip_path(self_path);
    if (!vcc_options.include_path) {
        vcc_options.include_path = format_string_interned(arena, "%s/../share/vcc/include/", working_dir);
    }
    growy_append_formatted(g, " -c -emit-llvm -S -g -O0 -ffreestanding -Wno-main-return-type -Xclang -fpreserve-vec3-type --target=spir64-unknown-unknown -isystem\"%s\" -D__SHADY__=1", vcc_options.include_path);
    free(working_dir);
    free(self_path);

    if (vcc_options.only_run_clang)
        growy_append_formatted(g, " -o %s", args.output_filename);
    else {
        if (!vcc_options.tmp_filename) {
            vcc_options.tmp_filename = alloca(33);
            vcc_options.tmp_filename[32] = '\0';
            uint32_t hash = 0;
            for (size_t i = 0; i < num_source_files; i++) {
                String filename = read_list(const char*, args.input_filenames)[i];
                hash ^= hash_murmur(filename, strlen(filename));
            }
            srand(hash);
            for (size_t i = 0; i < 32; i++) {
                vcc_options.tmp_filename[i] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % (10 + 26 * 2)];
            }
        }
        growy_append_formatted(g, " -o %s", vcc_options.tmp_filename);
    }

    for (size_t i = 0; i < num_source_files; i++) {
        String filename = read_list(const char*, args.input_filenames)[i];

        growy_append_string(g, " \"");
        growy_append_bytes(g, strlen(filename), filename);
        growy_append_string(g, "\"");
    }

    growy_append_bytes(g, 1, "\0");
    char* arg_string = growy_deconstruct(g);

    info_print("built command: %s\n", arg_string);

    FILE* stream = popen(arg_string, "r");
    free(arg_string);

    Growy* json_bytes = new_growy();
    while (true) {
        char buf[4096];
        int read = fread(buf, 1, sizeof(buf), stream);
        if (read == 0)
            break;
        growy_append_bytes(json_bytes, read, buf);
    }
    growy_append_string(json_bytes, "\0");
    char* llvm_result = growy_deconstruct(json_bytes);
    int clang_returned = pclose(stream);
    info_print("Clang returned %d and replied: \n%s", clang_returned, llvm_result);
    free(llvm_result);
    if (clang_returned)
        exit(ClangInvocationFailed);

    if (!vcc_options.only_run_clang) {
        size_t len;
        char* llvm_ir;
        if (!read_file(vcc_options.tmp_filename, &len, &llvm_ir))
            exit(InputFileIOError);
        driver_load_source_file(SrcLLVM, len, llvm_ir, mod);
        free(llvm_ir);

        if (vcc_options.delete_tmp_file)
            remove(vcc_options.tmp_filename);

        driver_compile(&args, mod);
    }

    info_print("Done\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
