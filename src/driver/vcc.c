#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "growy.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    if (entries_count_list(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    IrArena* arena = new_ir_arena(default_arena_config(&args.config));
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    int clang_retval = system("clang --version");
    if (clang_retval != 0)
    error("clang not present in path or otherwise broken (retval=%d)", clang_retval);

    Growy* g = new_growy();
    growy_append_string(g, "clang");
    growy_append_string(g, " -c -emit-llvm -S -g -O0 -Wno-main-return-type -Xclang -fpreserve-vec3-type");
    growy_append_string(g, " -o vcc_tmp.ll");

    size_t num_source_files = entries_count_list(args.input_filenames);
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
    if (clang_returned)
        return ClangInvocationFailed;

    size_t len;
    char* llvm_ir;
    if (!read_file("vcc_tmp.ll", &len, &llvm_ir))
        return InputFileIOError;
    driver_load_source_file(SrcLLVM, len, llvm_ir, mod);

    driver_compile(&args, mod);
    info_print("Done\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
