#include "internal_constants.h"

#include "portability.h"
#include "ir_private.h"

#include <string.h>

void generate_dummy_constants(SHADY_UNUSED CompilerConfig* config, Module* mod) {
    IrArena* arena = get_module_arena(mod);
#define X(constant_name, T, placeholder) \
    Node* constant_name##_var = constant(mod, singleton(annotation(arena, (Annotation) { .name = "RetainAfterSpecialization" })), T, #constant_name); \
    constant_name##_var->payload.constant.instruction = quote_helper(arena, singleton(placeholder));
    INTERNAL_CONSTANTS(X)
#undef X
}
