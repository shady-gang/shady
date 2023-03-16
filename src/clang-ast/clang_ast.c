#include "clang_ast.h"

#include <assert.h>
#include <string.h>

#include "log.h"

static void toplevel_to_shady(ClangAst* ast, json_object* decl) {
    assert(decl);
    const char* name = json_object_get_string(json_object_object_get(decl, "name"));
    const char* kind = json_object_get_string(json_object_object_get(decl, "kind"));
    assert(name && kind);
    printf("decl: %s\n", name);

    if (strcmp(kind, "TypedefDecl") == 0) {}
    else if (strcmp(kind, "FunctionDecl") == 0) {
        const char* mangled_name = json_object_get_string(json_object_object_get(decl, "mangledName"));
        printf("fn: %s\n", mangled_name);
        json_object* jtype = json_object_object_get(decl, "type");
        const char* qualified_type = json_object_get_string(json_object_object_get(jtype, "qualType"));
        const Type* fn_type = convert_qualtype(ast, qualified_type);
        dump_node(fn_type);
    }
    else {
        assert(false);
    }
}

void ast_to_shady(json_object* object, Module* mod) {
    ClangAst converter = {
        .arena = get_module_arena(mod),
        .mod = mod,
    };
    assert(object && json_object_is_type(object, json_type_object));
    json_object* kind = json_object_object_get(object, "kind");
    assert(kind && json_object_get_string(kind) && strcmp(json_object_get_string(kind), "TranslationUnitDecl") == 0);
    debug_print("Parsed root json object successfully\n");

    json_object* inner = json_object_object_get(object, "inner");
    assert(inner && json_object_get_type(inner) == json_type_array);
    int len = json_object_array_length(inner);
    for (size_t i = 0; i < len; i++) {
        toplevel_to_shady(&converter, json_object_array_get_idx(inner, i));
    }
}
