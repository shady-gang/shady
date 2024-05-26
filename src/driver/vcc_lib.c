#include "vcc/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "growy.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

uint32_t hash_murmur(const void* data, size_t size);

void cli_parse_vcc_args(VccConfig* options, int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        else if (strcmp(argv[i], "--vcc-keep-tmp-file") == 0) {
            argv[i] = NULL;
            options->delete_tmp_file = false;
            options->tmp_filename = format_string_new("vcc_tmp.ll");
            continue;
        } else if (strcmp(argv[i], "--vcc-include-path") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                error("Missing subgroup size name");
            if (options->include_path)
                free((void*) options->include_path);
            options->include_path = format_string_new("%s", argv[i]);
            continue;
        } else if (strcmp(argv[i], "--only-run-clang") == 0) {
            argv[i] = NULL;
            options->only_run_clang = true;
            continue;
        }
    }

    cli_pack_remaining_args(pargc, argv);
}

void vcc_check_clang() {
    int clang_retval = system("clang --version");
    if (clang_retval != 0)
        error("clang not present in path or otherwise broken (retval=%d)", clang_retval);
}

VccConfig vcc_init_config() {
    VccConfig vcc_config = {
        .only_run_clang = false,
    };

    String self_path = get_executable_location();
    String working_dir = strip_path(self_path);
    if (!vcc_config.include_path) {
        vcc_config.include_path = format_string_new("%s/../share/vcc/include/", working_dir);
    }
    free((void*) working_dir);
    free((void*) self_path);
    return vcc_config;
}

void destroy_vcc_options(VccConfig vcc_options) {
    if (vcc_options.include_path)
        free((void*) vcc_options.include_path);
    if (vcc_options.tmp_filename)
        free((void*) vcc_options.tmp_filename);
}

void vcc_run_clang(VccConfig* vcc_options, size_t num_source_files, String* input_filenames) {
    Growy* g = new_growy();
    growy_append_string(g, "clang");
    String self_path = get_executable_location();
    String working_dir = strip_path(self_path);
    growy_append_formatted(g, " -c -emit-llvm -S -g -O0 -ffreestanding -Wno-main-return-type -Xclang -fpreserve-vec3-type --target=spir64-unknown-unknown -isystem\"%s\" -D__SHADY__=1", vcc_options->include_path);
    free((void*) working_dir);
    free((void*) self_path);

    if (!vcc_options->tmp_filename) {
        if (vcc_options->only_run_clang) {
            error_print("Please provide an output filename.\n");
            error_die();
        }
        char* tmp_alloc;
        vcc_options->tmp_filename = tmp_alloc = malloc(33);
        tmp_alloc[32] = '\0';
        uint32_t hash = 0;
        for (size_t i = 0; i < num_source_files; i++) {
            String filename = input_filenames[i];
            hash ^= hash_murmur(filename, strlen(filename));
        }
        srand(hash);
        for (size_t i = 0; i < 32; i++) {
            tmp_alloc[i] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % (10 + 26 * 2)];
        }
    }
    growy_append_formatted(g, " -o %s", vcc_options->tmp_filename);

    for (size_t i = 0; i < num_source_files; i++) {
        String filename = input_filenames[i];

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
}

Module* vcc_parse_back_into_module(CompilerConfig* config, VccConfig* vcc_options, String module_name) {
    size_t len;
    char* llvm_ir;
    if (!read_file(vcc_options->tmp_filename, &len, &llvm_ir))
        exit(InputFileIOError);
    Module* mod;
    driver_load_source_file(config, SrcLLVM, len, llvm_ir, module_name, &mod);
    free(llvm_ir);

    if (vcc_options->delete_tmp_file)
        remove(vcc_options->tmp_filename);

    return mod;
}