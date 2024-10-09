#ifndef SHADY_IR_TYPE_H
#define SHADY_IR_TYPE_H

#include "shady/ir/grammar.h"

/// Unit type, carries no information (equivalent to C's void)
/// There is exactly one possible value of this type: ()
static inline const Node* unit_type(IrArena* arena) {
    return record_type(arena, (RecordType) {
        .members = shd_empty(arena),
    });
}

Type* nominal_type(Module*, Nodes annotations, String name);

const Type* shd_get_actual_mask_type(IrArena* arena);

bool shd_is_subtype(const Type* supertype, const Type* type);
void shd_check_subtype(const Type* supertype, const Type* type);

/// Is this a type that a value in the language can have ?
bool shd_is_value_type(const Type*);

/// Is this a valid data type (for usage in other types and as type arguments) ?
bool shd_is_data_type(const Type*);

bool shd_is_arithm_type(const Type*);
bool shd_is_shiftable_type(const Type*);
bool shd_has_boolean_ops(const Type*);
bool shd_is_comparable_type(const Type*);
bool shd_is_ordered_type(const Type*);
bool shd_is_physical_ptr_type(const Type* t);
bool shd_is_generic_ptr_type(const Type* t);

bool shd_is_reinterpret_cast_legal(const Type* src_type, const Type* dst_type);
bool shd_is_conversion_legal(const Type* src_type, const Type* dst_type);

/// Returns the (possibly qualified) pointee type from a (possibly qualified) ptr type
const Type* shd_get_pointee_type(IrArena*, const Type*);

String shd_get_address_space_name(AddressSpace);
/// Returns false iff pointers in that address space can contain different data at the same address
/// (amongst threads in the same subgroup)
bool shd_is_addr_space_uniform(IrArena*, AddressSpace);

String shd_get_type_name(IrArena* arena, const Type* t);

const Type* maybe_multiple_return(IrArena* arena, Nodes types);
Nodes unwrap_multiple_yield_types(IrArena* arena, const Type* type);

/// Collects the annotated types in the list of variables
/// NB: this is different from get_values_types, that function uses node.type, whereas this one uses node.payload.var.type
/// This means this function works in untyped modules where node.type is NULL.
Nodes shd_get_param_types(IrArena* arena, Nodes variables);

Nodes shd_get_values_types(IrArena*, Nodes);

// Qualified type helpers
/// Ensures an operand has divergence-annotated type and extracts it
const Type* shd_get_unqualified_type(const Type*);
bool shd_is_qualified_type_uniform(const Type*);
bool shd_deconstruct_qualified_type(const Type**);

const Type* shd_as_qualified_type(const Type* type, bool uniform);

Nodes shd_strip_qualifiers(IrArena*, Nodes);
Nodes shd_add_qualifiers(IrArena*, Nodes, bool);

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
AddressSpace deconstruct_pointer_type(const Type**);

// Nominal type helpers
const Node* get_nominal_type_decl(const Type*);
const Type* get_nominal_type_body(const Type*);
const Node* get_maybe_nominal_type_decl(const Type*);
const Type* get_maybe_nominal_type_body(const Type*);

// Composite type helpers
Nodes get_composite_type_element_types(const Type*);
const Node* get_fill_type_element_type(const Type*);
const Node* get_fill_type_size(const Type*);

#endif
