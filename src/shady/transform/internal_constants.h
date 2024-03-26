#ifndef SHADY_INTERNAL_CONSTANTS
#define SHADY_INTERNAL_CONSTANTS

#include "shady/ir.h"

#define INTERNAL_CONSTANTS(X) \
X(SUBGROUP_SIZE, int32_type(arena), uint32_literal(arena, config->specialization.subgroup_size)) \
X(SUBGROUPS_PER_WG, int32_type(arena), uint32_literal(arena, 1)) \

void generate_dummy_constants(CompilerConfig* config, Module*);

#endif
