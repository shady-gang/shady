#ifndef SHADY_IR_COMPOSITE_H
#define SHADY_IR_COMPOSITE_H

#include "shady/ir/grammar.h"

const Node* shd_maybe_tuple_helper(IrArena* a, Nodes values);
const Node* shd_tuple_helper(IrArena*, Nodes contents);

const Node* shd_extract_helper(IrArena* a, const Node* base, Nodes selectors);
const Node* shd_extract_single_helper(IrArena* a, const Node* composite, const Node* index);

void shd_enter_composite_type(const Type** datatype, bool* uniform, const Node* selector, bool allow_entering_pack);
void shd_enter_composite_type_indices(const Type** datatype, bool* uniform, Nodes indices, bool allow_entering_pack);

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count);

#endif
