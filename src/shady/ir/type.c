#include "../ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

const Type* maybe_multiple_return(IrArena* arena, Nodes types) {
    switch (types.count) {
        case 0: return empty_multiple_return_type(arena);
        case 1: return types.nodes[0];
        default: return record_type(arena, (RecordType) {
                .members = types,
                .names = shd_strings(arena, 0, NULL),
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
            return shd_singleton(type);
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

Nodes get_param_types(IrArena* arena, Nodes variables) {
    LARRAY(const Type*, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        assert(variables.nodes[i]->tag == Param_TAG);
        arr[i] = variables.nodes[i]->payload.param.type;
    }
    return shd_nodes(arena, variables.count, arr);
}

Nodes get_values_types(IrArena* arena, Nodes values) {
    assert(arena->config.check_types);
    LARRAY(const Type*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = values.nodes[i]->type;
    return shd_nodes(arena, values.count, arr);
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
    } else shd_error("Expected a value type (annotated with qual_type)")
}

const Type* shd_as_qualified_type(const Type* type, bool uniform) {
    return qualified_type(type->arena, (QualifiedType) { .type = type, .is_uniform = uniform });
}

Nodes strip_qualifiers(IrArena* arena, Nodes tys) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = get_unqualified_type(tys.nodes[i]);
    return shd_nodes(arena, tys.count, arr);
}

Nodes add_qualifiers(IrArena* arena, Nodes tys, bool uniform) {
    LARRAY(const Type*, arr, tys.count);
    for (size_t i = 0; i < tys.count; i++)
        arr[i] = shd_as_qualified_type(tys.nodes[i],
                                       uniform || !arena->config.is_simt /* SIMD arenas ban varying value types */);
    return shd_nodes(arena, tys.count, arr);
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
            size_t size = shd_get_int_literal_value(*shd_resolve_to_int_literal(get_fill_type_size(type)), false);
            if (size >= 1024) {
                shd_warn_print("Potential performance issue: creating a really big array of composites of types (size=%d)!\n", size);
            }
            const Type* element_type = get_fill_type_element_type(type);
            LARRAY(const Type*, types, size);
            for (size_t i = 0; i < size; i++) {
                types[i] = element_type;
            }
            return shd_nodes(type->arena, size, types);
        }
        default: shd_error("Not a composite type !")
    }
}

const Node* get_fill_type_element_type(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.element_type;
        case PackType_TAG: return composite_t->payload.pack_type.element_type;
        default: shd_error("fill values need to be either array or pack types")
    }
}

const Node* get_fill_type_size(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.size;
        case PackType_TAG: return shd_int32_literal(composite_t->arena, composite_t->payload.pack_type.width);
        default: shd_error("fill values need to be either array or pack types")
    }
}

Type* nominal_type(Module* mod, Nodes annotations, String name) {
    IrArena* arena = mod->arena;
    NominalType payload = {
        .name = string(arena, name),
        .module = mod,
        .annotations = annotations,
        .body = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = NominalType_TAG,
        .payload.nom_type = payload
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}
