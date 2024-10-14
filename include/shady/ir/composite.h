#ifndef SHADY_IR_COMPOSITE_H
#define SHADY_IR_COMPOSITE_H

#include "shady/ir/grammar.h"

const Node* maybe_tuple_helper(IrArena* a, Nodes values);
const Node* extract_helper(const Node* composite, const Node* index);

const Node* tuple_helper(IrArena*, Nodes contents);

void shd_enter_composite_type(const Type** datatype, bool* uniform, const Node* selector, bool allow_entering_pack);
void shd_enter_composite_type_indices(const Type** datatype, bool* uniform, Nodes indices, bool allow_entering_pack);

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count);

#endif
