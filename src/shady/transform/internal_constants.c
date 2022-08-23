#include "internal_constants.h"

#include <string.h>

Nodes generate_dummy_constants(CompilerConfig* config, IrArena* arena) {
#define X(name, placeholder, real) \
    Node* name##_var = constant(arena, nodes(arena, 0, NULL), #name); \
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

void patch_constants(CompilerConfig* config, IrArena* arena, Node* root) {
    Nodes decls = root->payload.root.declarations;
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