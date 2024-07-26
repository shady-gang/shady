#ifndef SHADY_IR_GEN_HELPERS_H
#define SHADY_IR_GEN_HELPERS_H

#include "shady/ir.h"
#include "shady/builtins.h"

void gen_push_value_stack(BodyBuilder* bb, const Node* value);
void gen_push_values_stack(BodyBuilder* bb, Nodes values);
const Node* gen_pop_value_stack(BodyBuilder*, const Type* type);

Nodes gen_call(BodyBuilder*, const Node* callee, Nodes args);
Nodes gen_primop(BodyBuilder*, Op, Nodes, Nodes);
Nodes gen_primop_c(BodyBuilder*, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_ce(BodyBuilder*, Op op, size_t operands_count, const Node* operands[]);
const Node* gen_primop_e(BodyBuilder*, Op op, Nodes, Nodes);

const Node* gen_reinterpret_cast(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_conversion(BodyBuilder*, const Type* dst, const Node* src);
const Node* gen_merge_halves(BodyBuilder*, const Node* lo, const Node* hi);

const Node* gen_stack_alloc(BodyBuilder*, const Type* ptr);
const Node* gen_local_alloc(BodyBuilder*, const Type* ptr);

const Node* gen_load(BodyBuilder*, const Node* ptr);
void gen_store(BodyBuilder*, const Node* ptr, const Node* value);
const Node* gen_lea(BodyBuilder*, const Node* base, const Node* offset, Nodes selectors);
const Node* gen_extract(BodyBuilder*, const Node* base, Nodes selectors);
void gen_comment(BodyBuilder*, String str);
const Node* get_builtin(Module* m, Builtin b);
const Node* get_or_create_builtin(Module* m, Builtin b, String n);
const Node* gen_builtin_load(Module*, BodyBuilder*, Builtin);

Nodes gen_if(BodyBuilder*, const Node*, Nodes, const Node*, const Node*);

typedef struct Rewriter_ Rewriter;

const Node* find_or_process_decl(Rewriter*, const char* name);
const Node* access_decl(Rewriter*, const char* name);

const Node* convert_int_extend_according_to_src_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_extend_according_to_dst_t(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_zero_extend(BodyBuilder*, const Type* dst_type, const Node* src);
const Node* convert_int_sign_extend(BodyBuilder*, const Type* dst_type, const Node* src);

const Node* get_default_zero_value(IrArena*, const Type*);

bool is_builtin_load_op(const Node*, Builtin*);

#endif
