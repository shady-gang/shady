#include "internal_constants.h"

#include "portability.h"
#include "ir_private.h"

#include <string.h>

void shd_generate_dummy_constants(SHADY_UNUSED const CompilerConfig* config, Module* mod) {
    IrArena* arena = shd_module_get_arena(mod);
    Nodes annotations = mk_nodes(arena, annotation_helper(arena, "Internal"), annotation_helper(arena, "Weak"));
#define X(constant_name, T, placeholder) \
    Node* constant_name##_var = constant(mod, annotations, T, #constant_name); \
    constant_name##_var->payload.constant.value = placeholder;
    INTERNAL_CONSTANTS(X)
#undef X
}
