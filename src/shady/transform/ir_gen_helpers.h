#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"

void gen_push_value_stack(BodyBuilder* instructions, const Node* value);
void gen_push_values_stack(BodyBuilder* instructions, Nodes values);
const Node* gen_pop_value_stack(BodyBuilder* instructions, const Type* type);

Nodes gen_primop(BodyBuilder*, Op, Nodes);
Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes);

const Node* gen_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src);
const Node* gen_merge_i32s_i64(BodyBuilder*, const Node* lo, const Node* hi);

const Node* gen_load(BodyBuilder*, const Node* ptr);
void gen_store(BodyBuilder*, const Node* ptr, const Node* value);
const Node* gen_lea(BodyBuilder*, const Node* base, const Node* offset, Nodes selectors);

typedef struct Rewriter_ Rewriter;

const Node* find_or_process_decl(Rewriter*, Module* mod, const char* name);
const Node* access_decl(Rewriter*, Module* mod, const char* name);

#endif
