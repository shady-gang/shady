#include "ir_private.h"
#include "type.h"
#include "log.h"
#include "portability.h"

#include <assert.h>

const Type* maybe_multiple_return(IrArena* arena, Nodes types) {
    switch (types.count) {
        case 0: return empty_multiple_return_type(arena);
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
        default:
            assert(is_value_type(type));
            return singleton(type);
    }
}

const Type* get_pointee_type(IrArena* arena, const Type* type) {
    bool qualified = false, uniform = false;
    if (is_value_type(type)) {
        qualified = true;
        uniform = is_qualified_type_uniform(type);
        type = get_unqualified_type(type);
    }
    assert(type->tag == PtrType_TAG);
    uniform &= is_addr_space_uniform(arena, type->payload.ptr_type.address_space);
    type = type->payload.ptr_type.pointed_type;

    if (qualified)
        type = qualified_type(arena, (QualifiedType) {
            .type = type,
            .is_uniform = uniform
        });
    return type;
}

void step_composite(const Type** datatype, bool* uniform, const Node* selector, bool allow_entering_pack) {
    const Type* current_type = *datatype;

    if (selector->arena->config.check_types) {
        const Type* selector_type = selector->type;
        bool selector_uniform = deconstruct_qualified_type(&selector_type);
        assert(selector_type->tag == Int_TAG && "selectors must be integers");
        *uniform &= selector_uniform;
    }

    try_again:
    switch (current_type->tag) {
        case RecordType_TAG: {
            size_t selector_value = get_int_literal_value(*resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.record_type.members.count);
            current_type = current_type->payload.record_type.members.nodes[selector_value];
            break;
        }
        case ArrType_TAG: {
            current_type = current_type->payload.arr_type.element_type;
            break;
        }
        case TypeDeclRef_TAG: {
            const Node* nom_decl = current_type->payload.type_decl_ref.decl;
            assert(nom_decl->tag == NominalType_TAG);
            current_type = nom_decl->payload.nom_type.body;
            goto try_again;
        }
        case PackType_TAG: {
            assert(allow_entering_pack);
            assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a pack type need to be constant");
            size_t selector_value = get_int_literal_value(*resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.pack_type.width);
            current_type = current_type->payload.pack_type.element_type;
            break;
        }
            // also remember to assert literals for the selectors !
        default: {
            log_string(ERROR, "Trying to enter non-composite type '");
            log_node(ERROR, current_type);
            log_string(ERROR, "' with selector '");
            log_node(ERROR, selector);
            log_string(ERROR, "'.");
            error("");
        }
    }
    *datatype = current_type;
}

void enter_composite(const Type** datatype, bool* uniform, Nodes indices, bool allow_entering_pack) {
    for(size_t i = 0; i < indices.count; i++) {
        const Node* selector = indices.nodes[i];
        step_composite(datatype, uniform, selector, allow_entering_pack);
    }
}

Nodes get_param_types(IrArena* arena, Nodes variables) {
    LARRAY(const Type*, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        assert(variables.nodes[i]->tag == Param_TAG);
        arr[i] = variables.nodes[i]->payload.param.type;
    }
    return nodes(arena, variables.count, arr);
}

Nodes get_values_types(IrArena* arena, Nodes values) {
    assert(arena->config.check_types);
    LARRAY(const Type*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = values.nodes[i]->type;
    return nodes(arena, values.count, arr);
}

bool is_qualified_type_uniform(const Type* type) {
    const Type* result_type = type;
    bool is_uniform = deconstruct_qualified_type(&result_type);
    return is_uniform;
}

const Type* get_unqualified_type(const Type* type) {
    assert(is_type(type));
    const Type* result_type = type;
    deconstruct_qualified_type(&result_type);
    return result_type;
}

bool deconstruct_qualified_type(const Type** type_out) {
    const Type* type = *type_out;
    if (type->tag == QualifiedType_TAG) {
        *type_out = type->payload.qualified_type.type;
        return type->payload.qualified_type.is_uniform;
    } else error("Expected a value type (annotated with qual_type)")
}

const Type* qualified_type_helper(const Type* type, bool uniform) {
    return qualified_type(type->arena, (QualifiedType) { .type = type, .is_uniform = uniform });
}

Nodes strip_qualifiers(IrArena* arena, Nodes tys) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = get_unqualified_type(tys.nodes[i]);
    return nodes(arena, tys.count, arr);
}

Nodes add_qualifiers(IrArena* arena, Nodes tys, bool uniform) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = qualified_type_helper(tys.nodes[i], uniform || !arena->config.is_simt /* SIMD arenas ban varying value types */);
    return nodes(arena, tys.count, arr);
}

const Type* get_packed_type_element(const Type* type) {
    const Type* t = type;
    deconstruct_packed_type(&t);
    return t;
}

size_t get_packed_type_width(const Type* type) {
    const Type* t = type;
    return deconstruct_packed_type(&t);
}

size_t deconstruct_packed_type(const Type** type) {
    assert((*type)->tag == PackType_TAG);
    return deconstruct_maybe_packed_type(type);
}

const Type* get_maybe_packed_type_element(const Type* type) {
    const Type* t = type;
    deconstruct_maybe_packed_type(&t);
    return t;
}

size_t get_maybe_packed_type_width(const Type* type) {
    const Type* t = type;
    return deconstruct_maybe_packed_type(&t);
}

size_t deconstruct_maybe_packed_type(const Type** type) {
    const Type* t = *type;
    assert(is_data_type(t));
    if (t->tag == PackType_TAG) {
        *type = t->payload.pack_type.element_type;
        return t->payload.pack_type.width;
    }
    return 1;
}

const Type* maybe_packed_type_helper(const Type* type, size_t width) {
    assert(width > 0);
    if (width == 1)
        return type;
    return pack_type(type->arena, (PackType) {
        .width = width,
        .element_type = type,
    });
}

const Type* get_pointer_type_element(const Type* type) {
    const Type* t = type;
    deconstruct_pointer_type(&t);
    return t;
}

AddressSpace deconstruct_pointer_type(const Type** type) {
    const Type* t = *type;
    assert(t->tag == PtrType_TAG);
    *type = t->payload.ptr_type.pointed_type;
    return t->payload.ptr_type.address_space;
}

const Node* get_nominal_type_decl(const Type* type) {
    assert(type->tag == TypeDeclRef_TAG);
    return get_maybe_nominal_type_decl(type);
}

const Type* get_nominal_type_body(const Type* type) {
    assert(type->tag == TypeDeclRef_TAG);
    return get_maybe_nominal_type_body(type);
}

const Node* get_maybe_nominal_type_decl(const Type* type) {
    if (type->tag == TypeDeclRef_TAG) {
        const Node* decl = type->payload.type_decl_ref.decl;
        assert(decl->tag == NominalType_TAG);
        return decl;
    }
    return NULL;
}

const Type* get_maybe_nominal_type_body(const Type* type) {
    const Node* decl = get_maybe_nominal_type_decl(type);
    if (decl)
        return decl->payload.nom_type.body;
    return type;
}

Nodes get_composite_type_element_types(const Type* type) {
    switch (is_type(type)) {
        case Type_TypeDeclRef_TAG: {
            type = get_nominal_type_body(type);
            assert(type->tag == RecordType_TAG);
            SHADY_FALLTHROUGH
        }
        case RecordType_TAG: {
            return type->payload.record_type.members;
        }
        case Type_ArrType_TAG:
        case Type_PackType_TAG: {
            size_t size = get_int_literal_value(*resolve_to_int_literal(get_fill_type_size(type)), false);
            if (size >= 1024) {
                warn_print("Potential performance issue: creating a really big array of composites of types (size=%d)!\n", size);
            }
            const Type* element_type = get_fill_type_element_type(type);
            LARRAY(const Type*, types, size);
            for (size_t i = 0; i < size; i++) {
                types[i] = element_type;
            }
            return nodes(type->arena, size, types);
        }
        default: error("Not a composite type !")
    }
}

const Node* get_fill_type_element_type(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.element_type;
        case PackType_TAG: return composite_t->payload.pack_type.element_type;
        default: error("fill values need to be either array or pack types")
    }
}

const Node* get_fill_type_size(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.size;
        case PackType_TAG: return int32_literal(composite_t->arena, composite_t->payload.pack_type.width);
        default: error("fill values need to be either array or pack types")
    }
}
