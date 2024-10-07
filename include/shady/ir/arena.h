#ifndef SHADY_IR_ARENA_H
#define SHADY_IR_ARENA_H

#include "shady/ir/base.h"
#include "shady/config.h"

IrArena* shd_new_ir_arena(const ArenaConfig* config);
void shd_destroy_ir_arena(IrArena* arena);
const Node* shd_get_node_by_id(const IrArena* a, NodeId id);

#endif
