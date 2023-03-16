#include "clang_ast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "log.h"
#include "portability.h"
#include "growy.h"

int main(int argc, char** argv) {
    int clang_retval = system("clang --version");
    if (clang_retval != 0)
        error("clang not present in path or otherwise broken (retval=%d)", clang_retval);

    Growy* g = new_growy();
    growy_append(g, "clang");
    growy_append(g, " -Xclang -ast-dump=json");
    growy_append(g, " -c");

    for (size_t i = 1; i < argc; i++) {
        growy_append(g, " \"");
        growy_append_bytes(g, strlen(argv[i]), argv[i]);
        growy_append(g, "\"");
    }

    growy_append(g, "\0");
    char* arg_string = growy_deconstruct(g);

    debug_print("built command: %s\n", arg_string);

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
    growy_append(json_bytes, "\0");
    char* json_string = growy_deconstruct(json_bytes);

    debugv_print("json: %s\n", json_string);
    json_object* root = json_tokener_parse(json_string);
    free(json_string);

    IrArena* arena = new_ir_arena(default_arena_config());
    Module* mod = new_module(arena, "my_module");

    ast_to_shady(root, mod);

    dump_module(mod);

    json_object_put(root);
    return 0;
}
