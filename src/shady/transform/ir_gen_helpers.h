#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"

void gen_push_value_stack(BodyBuilder* instructions, const Node* value);
void gen_push_values_stack(BodyBuilder* instructions, Nodes values);
void gen_push_fn_stack(BodyBuilder* instructions, const Node* fn_ptr);
const Node* gen_pop_fn_stack(BodyBuilder* instructions, String var_name);
const Node* gen_pop_value_stack(BodyBuilder* instructions, String var_name, const Type* type);
Nodes gen_pop_values_stack(BodyBuilder* instructions, String var_name, const Nodes types);

Nodes gen_primop(BodyBuilder*, Op, Nodes);
Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes);

const Node* gen_merge_i32s_i64(BodyBuilder*, const Node* lo, const Node* hi);

const Node* gen_load(BodyBuilder*, const Node* ptr);
void gen_store(BodyBuilder*, const Node* ptr, const Node* value);
const Node* gen_lea(BodyBuilder*, const Node* base, const Node* offset, Nodes selectors);

typedef struct Rewriter_ Rewriter;

const Node* find_or_process_decl(Rewriter*, const Node* root, const char* name);
const Node* access_decl(Rewriter*, const Node* root, const char* name);

#endif
