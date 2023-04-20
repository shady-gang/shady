#include "shady/cli.h"
#include "clang_ast.h"
#include "json-c/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "log.h"
#include "portability.h"
#include "growy.h"

void ast_to_shady(json_object* object, Module* mod);

void parse_c_file(const char* filename, Module* mod) {
    int clang_retval = system("clang --version");
    if (clang_retval != 0)
        error("clang not present in path or otherwise broken (retval=%d)", clang_retval);

    Growy* g = new_growy();
    growy_append_string_literal(g, "clang");
    growy_append_string_literal(g, " -Xclang -ast-dump=json");
    growy_append_string_literal(g, " -c");

    growy_append_string_literal(g, " \"");
    growy_append_bytes(g, strlen(filename), filename);
    growy_append_string_literal(g, "\"");

    growy_append_string_literal(g, "\0");
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
    growy_append_string_literal(json_bytes, "\0");
    char* json_string = growy_deconstruct(json_bytes);

    debugv_print("json: %s\n", json_string);
    json_tokener* tokener = json_tokener_new_ex(512);
    json_object* root = json_tokener_parse_ex(tokener, json_string, strlen(json_string));
    enum json_tokener_error err = json_tokener_get_error(tokener);
    if (err != json_tokener_success) {
        error("Json tokener error: %s\n", json_tokener_error_desc(err));
    }

    ast_to_shady(root, mod);
    free(json_string);

    // dump_module(mod);

    json_object_put(root);
    json_tokener_free(tokener);
}
