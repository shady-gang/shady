#ifndef SHADY_IR_TYPE_H
#define SHADY_IR_TYPE_H

#include "shady/ir/grammar.h"

const Type* shd_get_exec_mask_type(IrArena* arena);
const Node* shd_get_default_value(IrArena* a, const Type* t);

bool shd_is_subtype(const Type* supertype, const Type* type);
void shd_check_subtype(const Type* supertype, const Type* type);

/// Is this a type that a value in the language can have ?
bool shd_is_value_type(const Type*);

bool shd_is_data_type(const Type*);
bool shd_is_physical_data_type(const Type* t);

bool shd_is_arithm_type(const Type*);
bool shd_is_shiftable_type(const Type*);
bool shd_has_boolean_ops(const Type*);
bool shd_is_comparable_type(const Type*);
bool shd_is_ordered_type(const Type*);
bool shd_is_physical_ptr_type(const Type* t);
bool shd_is_generic_ptr_type(const Type* t);

String shd_get_address_space_name(AddressSpace);

/// Returns the scope of an address space.
/// Two identical pointers are considered pointing to the same data from the perspective of two different invocations if they're on the same instance of the scope.
/// Examples:
/// AsPrivate has scope Invocation, two threads with the same pointer load different data.
/// AsShared has scope Workgroup, two threads in the same workgroup will see the same data.
/// AsGlobal has scope Device
/// AsGeneric can be any of AsPrivate | AsShared | AsGlobal so it gets the maximum, Invocation scope
/// TODO: CrossDevice isn't currently used. Evaluate at a later date.
ShdScope shd_get_addr_space_scope(AddressSpace);

String shd_get_type_name(IrArena* arena, const Type* t);

const Type* shd_maybe_multiple_return(IrArena* arena, Nodes types);
Nodes shd_unwrap_multiple_yield_types(IrArena* arena, const Type* type);

/// Collects the annotated types in the list of variables
/// NB: this is different from get_values_types, that function uses node.type, whereas this one uses node.payload.var.type
/// This means this function works in untyped modules where node.type is NULL.
Nodes shd_get_param_types(IrArena* arena, Nodes variables);

Nodes shd_get_values_types(IrArena*, Nodes);

// Qualified type helpers
/// Ensures an operand has divergence-annotated type and extracts it
const Type* shd_get_unqualified_type(const Type*);
ShdScope shd_get_qualified_type_scope(const Type*);
ShdScope shd_deconstruct_qualified_type(const Type**);

ShdScope shd_combine_scopes(ShdScope, ShdScope);

Nodes shd_strip_qualifiers(IrArena*, Nodes);
Nodes shd_add_qualifiers(IrArena*, Nodes, ShdScope);

// vector type helpers
const Type* shd_get_vector_type_element(const Type* type);
size_t shd_get_vector_type_width(const Type* type);
size_t shd_deconstruct_vector_type(const Type** type);

/// Helper for creating vector types, wraps the type in a vector_type if width > 1
const Type* shd_maybe_vector_type_helper(const Type* type, size_t width);

/// 'Maybe' variants that work with any types, and assume width=1 for non-vector types
/// Useful for writing generic type checking code !
const Type* shd_get_maybe_vector_type_element(const Type* type);
size_t shd_get_maybe_vector_type_width(const Type* type);
size_t shd_deconstruct_maybe_vector_type(const Type** type);

// Pointer type helpers
const Type* shd_get_pointer_type_element(const Type* type);
AddressSpace shd_deconstruct_pointer_type(const Type** type);

// Composite type helpers
Nodes shd_get_composite_type_element_types(const Type* type);
const Node* shd_get_fill_type_element_type(const Type* composite_t);
const Node* shd_get_fill_type_size(const Type* composite_t);

#endif
