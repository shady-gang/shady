#ifndef SHADY_INTERNAL_CONSTANTS
#define SHADY_INTERNAL_CONSTANTS

#include "shady/ir.h"

#define INTERNAL_CONSTANTS(X) \
X(SUBGROUP_SIZE, int32_literal(arena, 0), int32_literal(arena, (int32_t) config->subgroup_size))

Nodes generate_dummy_constants(CompilerConfig* config, IrArena* arena);
void patch_constants(CompilerConfig* config, IrArena* arena, Node* root);

#endif
