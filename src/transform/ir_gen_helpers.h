#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"

void gen_push_value_stack(BlockBuilder* instructions, const Node* value);
void gen_push_values_stack(BlockBuilder* instructions, Nodes values);
void gen_push_fn_stack(BlockBuilder* instructions, const Node* fn_ptr);
const Node* gen_pop_fn_stack(BlockBuilder* instructions, String var_name);
const Node* gen_pop_value_stack(BlockBuilder* instructions, String var_name, const Type* type);
Nodes gen_pop_values_stack(BlockBuilder* instructions, String var_name, const Nodes types);

Nodes gen_primop(BlockBuilder*, PrimOp);
const Node* gen_load(BlockBuilder*, const Node* ptr);
void gen_store(BlockBuilder*, const Node* ptr, const Node* value);
const Node* gen_lea(BlockBuilder*, const Node* base, const Node* offset, Nodes selectors);

#endif
