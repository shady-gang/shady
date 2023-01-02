#ifndef SHADY_TYPE_H
#define SHADY_TYPE_H

#include "shady/ir.h"

const Type* noret_type(IrArena* arena);

bool is_subtype(const Type* supertype, const Type* type);
void check_subtype(const Type* supertype, const Type* type);

/// Is this a type that a value in the language can have ?
bool is_value_type(const Type*);

/// Is this a valid data type (for usage in other types and as type arguments) ?
bool is_data_type(const Type*);

#define DEFINE_NODE_CHECK_FN_1_1(struct_name, short_name) const Type* check_type_##short_name(IrArena*, struct_name);
#define DEFINE_NODE_CHECK_FN_1_0(struct_name, short_name) const Type* check_type_##short_name(IrArena*);

#define DEFINE_NODE_CHECK_FN_1(struct_name, short_name, has_payload) DEFINE_NODE_CHECK_FN_1_##has_payload(struct_name, short_name)
#define DEFINE_NODE_CHECK_FN_0(struct_name, short_name, _)
#define DEFINE_NODE_CHECK_FN(_, has_typing_fn, has_payload, struct_name, short_name) DEFINE_NODE_CHECK_FN_##has_typing_fn(struct_name, short_name, has_payload)
NODES(DEFINE_NODE_CHECK_FN)
#undef DEFINE_NODE_CHECK_FN

const Type* get_actual_mask_type(IrArena* arena);

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

// Qualified type helpers
/// Ensures an operand has divergence-annotated type and extracts it
const Type* get_unqualified_type(const Type*);
bool is_qualified_type_uniform(const Type*);
bool deconstruct_qualified_type(const Type**);

const Type* qualified_type_helper(const Type*, bool uniform);
bool contains_qualified_type(const Type*);

Nodes strip_qualifiers(IrArena*, Nodes);
Nodes add_qualifiers(IrArena*, Nodes, bool);

// Pack (vector) type helpers
const Type* get_packed_type_element(const Type*);
size_t get_packed_type_width(const Type*);
size_t deconstruct_packed_type(const Type**);

/// Helper for creating pack types, wraps type in a pack_type if width > 1
const Type* maybe_packed_type_helper(const Type*, size_t width);

/// 'Maybe' variants that work with any types, and assume width=1 for non-packed types
/// Useful for writing generic type checking code !
const Type* get_maybe_packed_type_element(const Type*);
size_t get_maybe_packed_type_width(const Type*);
size_t deconstruct_maybe_packed_type(const Type**);

// Pointer type helpers
const Type* get_pointer_type_element(const Type*);
AddressSpace get_pointer_type_address_space(const Type*);
AddressSpace deconstruct_pointer_type(const Type**);

// Nominal type helpers
const Node* get_nominal_type_decl(const Type*);
const Type* get_nominal_type_body(const Type*);
const Node* get_maybe_nominal_type_decl(const Type*);
const Type* get_maybe_nominal_type_body(const Type*);

// Composite type helpers
Nodes get_composite_type_element_types(const Type*);

#endif
