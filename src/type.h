#ifndef SHADY_TYPE_H
#define SHADY_TYPE_H

#include "ir.h"

struct TypeTable;
struct TypeTable* new_type_table();
void destroy_type_table(struct TypeTable*);

bool is_subtype(const Type* supertype, const Type* type);
void check_subtype(const Type* supertype, const Type* type);

#define DEFINE_NODE_CHECK_FN_1(struct_name, short_name) Nodes check_type_##short_name(IrArena*, struct_name);
#define DEFINE_NODE_CHECK_FN_0(struct_name, short_name)
#define NODEDEF(_, has_typing_fn, _2, struct_name, short_name) DEFINE_NODE_CHECK_FN_##has_typing_fn(struct_name, short_name)
NODES()
#undef NODEDEF

Nodes op_params(IrArena* arena, Op op);
Nodes op_yields(IrArena* arena, Op op);

const Type* ensure_value_t(Nodes yields);

const Type* noret_type(IrArena* arena);

const Type* derive_fn_type(IrArena* arena, const Function* fn);

const Type* strip_qualifier(const Type* type, DivergenceQualifier* qual_out);
DivergenceQualifier get_qualifier(const Type* type);

#endif
