#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"

void gen_push_value_stack(BodyBuilder* bb, const Node* value);
void gen_push_values_stack(BodyBuilder* bb, Nodes values);
const Node* gen_pop_value_stack(BodyBuilder* instructions, const Type* type);

Nodes gen_primop(BodyBuilder*, Op, Nodes, Nodes);
Nodes gen_primop_c(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_ce(BodyBuilder* bb, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_e(BodyBuilder* bb, Op op, Nodes, Nodes);

const Node* gen_reinterpret_cast(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_conversion(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_merge_halves(BodyBuilder*, const Node* lo, const Node* hi);

const Node* gen_load(BodyBuilder*, const Node* ptr);
void gen_store(BodyBuilder*, const Node* ptr, const Node* value);
const Node* gen_lea(BodyBuilder*, const Node* base, const Node* offset, Nodes selectors);
const Node* gen_extract(BodyBuilder*, const Node* base, Nodes selectors);
void gen_comment(BodyBuilder*, String str);

typedef struct Rewriter_ Rewriter;

const Node* find_or_process_decl(Rewriter*, const char* name);
const Node* access_decl(Rewriter*, const char* name);

const Node* convert_int_extend_according_to_src_t(BodyBuilder* bb, const Type* dst_type, const Node* src);
const Node* convert_int_extend_according_to_dst_t(BodyBuilder* bb, const Type* dst_type, const Node* src);

const Node* get_default_zero_value(IrArena*, const Type*);

#endif
