#ifndef SHADY_INTERNAL_CONSTANTS
#define SHADY_INTERNAL_CONSTANTS

#include "shady/ir.h"

#define INTERNAL_CONSTANTS(X) \
X(SUBGROUP_SIZE, shd_uint32_type(arena), shd_uint32_literal(arena, 64)) \
X(SUBGROUPS_PER_WG, shd_uint32_type(arena), shd_uint32_literal(arena, 1)) \

typedef struct CompilerConfig_ CompilerConfig;
void shd_generate_dummy_constants(const CompilerConfig* config, Module* mod);

#endif
