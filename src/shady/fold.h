#ifndef SHADY_FOLD_H
#define SHADY_FOLD_H

#include "shady/ir.h"

const Node* fold_node(IrArena* arena, const Node* instruction);
const Node* resolve_known_vars(const Node* node, bool stop_at_values);

#endif
