#ifndef SHADY_IR_COMPOSITE_H
#define SHADY_IR_COMPOSITE_H

#include "shady/ir/grammar.h"

const Node* maybe_tuple_helper(IrArena* a, Nodes values);
const Node* extract_helper(const Node* composite, const Node* index);

const Node* tuple_helper(IrArena*, Nodes contents);

void enter_composite(const Type** datatype, bool* uniform, const Node* selector, bool allow_entering_pack);
void enter_composite_indices(const Type** datatype, bool* uniform, Nodes indices, bool allow_entering_pack);

#endif
