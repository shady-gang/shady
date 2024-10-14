#ifndef SHADY_IR_STACK_H
#define SHADY_IR_STACK_H

#include "shady/ir/base.h"
#include "shady/ir/builder.h"

void shd_bld_stack_push_value(BodyBuilder* bb, const Node* value);
void shd_bld_stack_push_values(BodyBuilder* bb, Nodes values);
const Node* shd_bld_stack_pop_value(BodyBuilder* bb, const Type* type);
const Node* shd_bld_get_stack_base_addr(BodyBuilder* bb);
const Node* shd_bld_get_stack_size(BodyBuilder* bb);
void shd_bld_set_stack_size(BodyBuilder* bb, const Node* new_size);

#endif
