#include "type.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

bool is_operand_uniform(const Type* type) {
    const Type* result_type = type;
    bool is_uniform = deconstruct_qual_type(&result_type);
    return is_uniform;
}

const Type* get_operand_type(const Type* type) {
    const Type* result_type = type;
    deconstruct_qual_type(&result_type);
    return result_type;
}

bool deconstruct_qual_type(const Type** type_out) {
    const Type* type = *type_out;
    if (type->tag == QualifiedType_TAG) {
        *type_out = type->payload.qualified_type.type;
        return type->payload.qualified_type.is_uniform;
    } else error("Expected a value type (annotated with qual_type)")
}

const Type* qual_type_helper(const Type* type, bool uniform) {
    return qualified_type(type->arena, (QualifiedType) { .type = type, .is_uniform = uniform });
}

const Type* get_vector_element(const Type* type) {
    const Type* t = type;
    deconstruct_vector_size(&t);
    return t;
}

size_t get_vector_size(const Type* type) {
    const Type* t = type;
    return deconstruct_vector_size(&t);
}

size_t deconstruct_vector_size(const Type** type) {
    assert((*type)->tag == PackType_TAG);
    return deconstruct_maybe_vector_size(type);
}

const Type* get_maybe_vector_element(const Type* type) {
    const Type* t = type;
    deconstruct_maybe_vector_size(&t);
    return t;
}

size_t get_maybe_vector_size(const Type* type) {
    const Type* t = type;
    return deconstruct_maybe_vector_size(&t);
}

size_t deconstruct_maybe_vector_size(const Type** type) {
    const Type* t = *type;
    assert(!contains_qualified_type(t));
    if (t->tag == PackType_TAG) {
        *type = t->payload.pack_type.element_type;
        return t->payload.pack_type.width;
    }
    return 1;
}

const Type* maybe_pack_type_helper(const Type* type, size_t width) {
    assert(width > 0);
    if (width == 1)
        return type;
    return pack_type(type->arena, (PackType) {
        .width = width,
        .element_type = type,
    });
}

// TODO: this isn't really accurate to what we want...
// It would be better to have verify_is_value_type, verify_is_operand etc functions.
bool contains_qualified_type(const Type* type) {
    switch (type->tag) {
        case QualifiedType_TAG: return true;
        default: return false;
    }
}

const Type* get_pointee_type(IrArena* arena, const Type* type) {
    bool qualified = false, uniform = false;
    if (contains_qualified_type(type)) {
        qualified = true;
        uniform = is_operand_uniform(type);
        type = get_operand_type(type);
    }
    assert(type->tag == PtrType_TAG);
    uniform &= is_addr_space_uniform(type->payload.ptr_type.address_space);
    type = type->payload.ptr_type.pointed_type;
    if (qualified)
        type = qualified_type(arena, (QualifiedType) {
            .type = type,
            .is_uniform = uniform
        });
    return type;
}

Nodes get_variables_types(IrArena* arena, Nodes variables) {
    LARRAY(const Type*, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        assert(variables.nodes[i]->tag == Variable_TAG);
        arr[i] = variables.nodes[i]->payload.var.type;
    }
    return nodes(arena, variables.count, arr);
}

Strings get_variable_names(IrArena* arena, Nodes variables) {
    LARRAY(String, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++)
        arr[i] = variables.nodes[i]->payload.var.name;
    return strings(arena, variables.count, arr);
}

Nodes get_values_types(IrArena* arena, Nodes values) {
    LARRAY(const Type*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = values.nodes[i]->type;
    return nodes(arena, values.count, arr);
}

Nodes strip_qualifiers(IrArena* arena, Nodes tys) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = get_operand_type(tys.nodes[i]);
    return nodes(arena, tys.count, arr);
}

const Type* wrap_multiple_yield_types(IrArena* arena, Nodes types) {
    switch (types.count) {
        case 0: return unit_type(arena);
        case 1: return types.nodes[0];
        default: return record_type(arena, (RecordType) {
            .members = types,
            .names = strings(arena, 0, NULL),
            .special = MultipleReturn,
        });
    }
    SHADY_UNREACHABLE;
}

Nodes unwrap_multiple_yield_types(IrArena* arena, const Type* type) {
    switch (type->tag) {
        case RecordType_TAG:
            if (type->payload.record_type.special == MultipleReturn)
                return type->payload.record_type.members;
            // fallthrough
        default: return nodes(arena, 1, (const Node* []) { type });
    }
}