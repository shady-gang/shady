#ifndef SHADY_IR_COMPOSITE_H
#define SHADY_IR_COMPOSITE_H

#include "shady/ir/grammar.h"

const Node* shd_maybe_tuple_helper(IrArena* a, Nodes values);
const Node* shd_tuple_helper(IrArena*, Nodes contents);

const Node* shd_extract_helper(IrArena* a, const Node* base, Nodes selectors);
const Node* shd_extract_literal(IrArena* a, const Node* base, uint32_t selector);
const Node* shd_insert_helper(IrArena* a, const Node* base, Nodes selectors, const Node* replacement);

void shd_enter_composite_type(const Type** datatype, ShdScope* uniform, const Node* selector);
void shd_enter_composite_type_indices(const Type** datatype, ShdScope* uniform, Nodes indices);

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count);

#endif
