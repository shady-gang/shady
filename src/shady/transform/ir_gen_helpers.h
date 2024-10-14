#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"
#include "shady/ir/builtin.h"

void gen_push_value_stack(BodyBuilder* bb, const Node* value);
void gen_push_values_stack(BodyBuilder* bb, Nodes values);
const Node* gen_pop_value_stack(BodyBuilder*, const Type* type);
const Node* gen_get_stack_base_addr(BodyBuilder*);
const Node* gen_get_stack_size(BodyBuilder*);
void gen_set_stack_size(BodyBuilder*, const Node* new_size);

Nodes gen_call(BodyBuilder*, const Node* callee, Nodes args);
const Node* gen_primop_e(BodyBuilder*, Op op, Nodes, Nodes);
const Node* gen_ext_instruction(BodyBuilder*, String set, int opcode, const Type* return_t, Nodes operands);

const Node* gen_reinterpret_cast(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_conversion(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_merge_halves(BodyBuilder*, const Node* lo, const Node* hi);

void gen_comment(BodyBuilder*, String str);
void gen_debug_printf(BodyBuilder*, String pattern, Nodes args);

const Node* convert_int_extend_according_to_src_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_extend_according_to_dst_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_zero_extend(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_sign_extend(BodyBuilder*, const Type* dst_type, const Node* src);

#endif
