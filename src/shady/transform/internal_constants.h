#ifndef SHADY_INTERNAL_CONSTANTS
#define SHADY_INTERNAL_CONSTANTS

#include "shady/ir.h"

#define INTERNAL_CONSTANTS(X) \
X(SUBGROUP_SIZE, int32_type(arena), uint32_literal(arena, 0), uint32_literal(arena, (int32_t) config->subgroup_size))

Nodes generate_dummy_constants(CompilerConfig* config, Module*);
void patch_constants(CompilerConfig* config, Module*);

#endif
