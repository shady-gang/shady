#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"
#include "shady/ir/builtin.h"

Nodes gen_call(BodyBuilder*, const Node* callee, Nodes args);
const Node* gen_ext_instruction(BodyBuilder*, String set, int opcode, const Type* return_t, Nodes operands);

const Node* convert_int_extend_according_to_src_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_extend_according_to_dst_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_zero_extend(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_sign_extend(BodyBuilder*, const Type* dst_type, const Node* src);

#endif
