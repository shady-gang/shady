#include "internal_constants.h"

#include "portability.h"

#include <string.h>

void generate_dummy_constants(SHADY_UNUSED CompilerConfig* config, Module* mod) {
    IrArena* arena = get_module_arena(mod);
#define X(name, T, placeholder) \
    Node* name##_var = constant(mod, nodes(arena, 0, NULL), T, #name); \
    name##_var->payload.constant.value = placeholder;
    INTERNAL_CONSTANTS(X)
#undef X
}
