#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"

typedef struct {
    IrArena* arena;
    struct List* list;
} Instructions;

Instructions begin_instructions(IrArena*);
void copy_instructions(Instructions, Nodes);
void append_instr(Instructions, const Node*);
Nodes finish_instructions(Instructions);

const Node* wrap_in_let(IrArena*, const Node*);

void gen_push_value_stack(Instructions instructions, const Node* value);
void gen_push_values_stack(Instructions instructions, Nodes values);
void gen_push_fn_stack(Instructions instructions, const Node* fn_ptr);
const Node* gen_pop_fn_stack(Instructions instructions, String var_name);
const Node* gen_pop_value_stack(Instructions instructions, String var_name, const Type* type);
Nodes gen_pop_values_stack(Instructions instructions, String var_name, const Nodes types);

#endif
