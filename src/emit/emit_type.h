#ifndef SHADY_EMIT_TYPE_H
#define SHADY_EMIT_TYPE_H

#include "emit.h"

// SPIR-V doesn't have multiple return types, this bridges the gap...
SpvId nodes_to_codom(Emitter* emitter, Nodes return_types);
const Type* normalize_type(Emitter* emitter, const Type* type);

#endif
