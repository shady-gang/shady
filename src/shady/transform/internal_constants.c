#include "internal_constants.h"

#include "portability.h"

#include <string.h>

Nodes generate_dummy_constants(SHADY_UNUSED CompilerConfig* config, Module* mod) {
    IrArena* arena = get_module_arena(mod);
#define X(name, placeholder, real) \
    Node* name##_var = constant(mod, nodes(arena, 0, NULL), #name); \
    name##_var->payload.constant.value = placeholder;
    INTERNAL_CONSTANTS(X)
#undef X
#define X(name, placeholder, real) name##_var,
    const Node* constants[] = {
            INTERNAL_CONSTANTS(X)
    };
#undef X
    return nodes(arena, sizeof(constants) / sizeof(const Node*), constants);
}

void patch_constants(CompilerConfig* config, Module* mod) {
    IrArena* arena = get_module_arena(mod);
    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        Node* decl = (Node*) decls.nodes[i];
        if (decl->tag != Constant_TAG) continue;
#define X(name, placeholder, real) \
        if (strcmp(get_decl_name(decl), #name) == 0) \
            decl->payload.constant.value = real;
        INTERNAL_CONSTANTS(X)
#undef X
    }
}
