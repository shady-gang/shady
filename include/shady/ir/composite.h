#ifndef SHADY_IR_COMPOSITE_H
#define SHADY_IR_COMPOSITE_H

#include "shady/ir/grammar.h"

/// Unit type, carries no information (equivalent to C's void)
/// There is exactly one possible value of this type: ()
static inline const Node* unit_type(IrArena* arena) {
    return tuple_type(arena, (TupleType) {
        .members = shd_empty(arena),
    });
}

const Node* shd_maybe_tuple_helper(IrArena* a, Nodes values);
const Node* shd_tuple_helper(IrArena*, Nodes contents);

const Node* shd_extract_helper(IrArena* a, const Node* base, Nodes selectors);
const Node* shd_extract_literal(IrArena* a, const Node* base, uint32_t selector);
const Node* shd_insert_helper(IrArena* a, const Node* base, Nodes selectors, const Node* replacement);

void shd_enter_composite_type(const Type** datatype, ShdScope* uniform, const Node* selector);
void shd_enter_composite_type_indices(const Type** datatype, ShdScope* uniform, Nodes indices);

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count);

Type* shd_struct_type(IrArena* a, StructType payload);
static inline Type* struct_type(IrArena* a, StructType payload) { return shd_struct_type(a, payload); }
static inline Type* struct_type_helper(IrArena* a, ShdStructFlags flags) { return struct_type(a, (StructType) { .flags = flags }); }

String shd_get_struct_type_field_name(const Type* t, size_t i);
void shd_set_struct_type_field_name(Type* t, size_t i, String name);

// helper functions to create non-recursive structs
const Node* shd_struct_type_with_members(IrArena*, ShdStructFlags flags, Nodes member_types);
const Node* shd_struct_type_with_members_named(IrArena*, ShdStructFlags flags, Nodes member_types, Strings member_names);

// call this to finish two-part struct definitions start with struct_type(flag)
void shd_struct_type_set_members(Node*, Nodes member_types);
void shd_struct_type_set_members_named(Node*, Nodes member_types, Strings member_names);

#endif
