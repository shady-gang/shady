#ifndef SHADY_TYPE_H
#define SHADY_TYPE_H

#include "shady/ir.h"

struct TypeTable;
struct TypeTable* new_type_table();
void destroy_type_table(struct TypeTable*);

const Type* noret_type(IrArena* arena);

bool is_subtype(const Type* supertype, const Type* type);
void check_subtype(const Type* supertype, const Type* type);

#define DEFINE_NODE_CHECK_FN_1_1(struct_name, short_name) const Type* check_type_##short_name(IrArena*, struct_name);
#define DEFINE_NODE_CHECK_FN_1_0(struct_name, short_name) const Type* check_type_##short_name(IrArena*);

#define DEFINE_NODE_CHECK_FN_1(struct_name, short_name, has_payload) DEFINE_NODE_CHECK_FN_1_##has_payload(struct_name, short_name)
#define DEFINE_NODE_CHECK_FN_0(struct_name, short_name, _)
#define DEFINE_NODE_CHECK_FN(_, has_typing_fn, has_payload, struct_name, short_name) DEFINE_NODE_CHECK_FN_##has_typing_fn(struct_name, short_name, has_payload)
NODES(DEFINE_NODE_CHECK_FN)
#undef DEFINE_NODE_CHECK_FN

const Type* wrap_multiple_yield_types(IrArena* arena, Nodes types);
Nodes unwrap_multiple_yield_types(IrArena* arena, const Type* type);

/// Returns the (possibly qualified) pointee type from a (possibly qualified) ptr type
const Type* get_pointee_type(IrArena*, const Type*);

/// Collects the annotated types in the list of variables
/// NB: this is different from get_values_types, that function uses node.type, whereas this one uses node.payload.var.type
/// This means this function works in untyped modules where node.type is NULL.
Nodes get_variables_types(IrArena*, Nodes);
Strings get_variable_names(IrArena*, Nodes);

Nodes get_values_types(IrArena*, Nodes);
Nodes strip_qualifiers(IrArena*, Nodes);

/// Ensures an operand has divergence-annotated type and extracts it
const Type* get_operand_type(const Type*);
bool is_operand_uniform(const Type*);
void deconstruct_operand_type(const Type**, bool* is_uniform_out);

bool contains_qualified_type(const Type* type);

#endif
