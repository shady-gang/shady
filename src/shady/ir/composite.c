#include "shady/ir/composite.h"

#include "ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

const Node* shd_extract_helper(IrArena* a, const Node* base, Nodes selectors) {
    const Node* r = base;
    for (size_t i = 0; i < selectors.count; i++)
        r = extract_helper(a, r, selectors.nodes[i]);
    return r;
}

const Node* shd_insert_helper(IrArena* a, const Node* base, Nodes selectors, const Node* replacement) {
    const Node* r = replacement;
    for (size_t i = selectors.count - 1; i < selectors.count; i--)
        r = insert_helper(a, shd_extract_helper(a, base, shd_nodes(a, i, selectors.nodes)), selectors.nodes[i], r);
    return r;
}

const Node* shd_extract_literal(IrArena* a, const Node* base, uint32_t selector) {
    return extract_helper(a, base, shd_uint32_literal(a, selector));
}

const Node* shd_maybe_tuple_helper(IrArena* a, Nodes values) {
    if (values.count == 1)
        return shd_first(values);
    return shd_tuple_helper(a, values);
}

const Node* shd_tuple_helper(IrArena* a, Nodes contents) {
    const Type* t = NULL;
    //if (a->config.check_types) {
    //    // infer the type of the tuple
    //    Nodes member_types = shd_get_values_types(a, contents);
    //    t = tuple_type(a, (TupleType) {.members = shd_strip_qualifiers(a, member_types)});
    //}

    return composite_helper(a, NULL, contents);
}

Nodes shd_get_composite_type_element_types(const Type* type) {
    switch (is_type(type)) {
        case Type_StructType_TAG: {
            return type->payload.struct_type.members;
        }
        case Type_ArrType_TAG:
        case Type_VectorType_TAG:
        case Type_MatrixType_TAG: {
            size_t size = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_fill_type_size(type)), false);
            if (size >= 1024) {
                shd_warn_print("Potential performance issue: creating a really big array of composites of types (size=%d)!\n", size);
            }
            const Type* element_type = shd_get_fill_type_element_type(type);
            LARRAY(const Type*, types, size);
            for (size_t i = 0; i < size; i++) {
                types[i] = element_type;
            }
            return shd_nodes(type->arena, size, types);
        }
        default: shd_error("Not a composite type !")
    }
}

const Node* shd_get_fill_type_element_type(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.element_type;
        case VectorType_TAG: return composite_t->payload.vector_type.element_type;
        case MatrixType_TAG: return composite_t->payload.matrix_type.element_type;
        default: shd_error("fill values need to be either array or vector or matrix types")
    }
}

const Node* shd_get_fill_type_size(const Type* composite_t) {
    switch (composite_t->tag) {
        case ArrType_TAG: return composite_t->payload.arr_type.size;
        case VectorType_TAG: return shd_int32_literal(composite_t->arena, composite_t->payload.vector_type.width);
        case MatrixType_TAG: return shd_int32_literal(composite_t->arena, composite_t->payload.matrix_type.columns);
        default: shd_error("fill values need to be either array or vector or matrix types")
    }
}

void shd_enter_composite_type(const Type** datatype, ShdScope* scope, const Node* selector) {
    const Type* current_type = *datatype;

    if (selector->arena->config.check_types) {
        const Type* selector_type = selector->type;
        ShdScope selector_scope = shd_deconstruct_qualified_type(&selector_type);
        assert(selector_type->tag == Int_TAG && "selectors must be integers");
        *scope = shd_combine_scopes(*scope, selector_scope);
    }

    try_again:
    switch (current_type->tag) {
        case StructType_TAG: {
            size_t selector_value = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.struct_type.members.count);
            current_type = current_type->payload.struct_type.members.nodes[selector_value];
            break;
        }
        case ArrType_TAG: {
            current_type = current_type->payload.arr_type.element_type;
            break;
        }
        case VectorType_TAG: {
            assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a pack type need to be constant");
            size_t selector_value = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.vector_type.width);
            current_type = current_type->payload.vector_type.element_type;
            break;
        }
        case MatrixType_TAG: {
            assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a pack type need to be constant");
            size_t selector_value = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.vector_type.width);
            current_type = current_type->payload.vector_type.element_type;
            break;
        }
        // also remember to assert literals for the selectors !
        default: {
            shd_log_fmt(ERROR, "Trying to enter non-composite type '");
            shd_log_node(ERROR, current_type);
            shd_log_fmt(ERROR, "' with selector '");
            shd_log_node(ERROR, selector);
            shd_log_fmt(ERROR, "'.");
            shd_error("");
        }
    }
    *datatype = current_type;
}

void shd_enter_composite_type_indices(const Type** datatype, ShdScope* s, Nodes indices) {
    for(size_t i = 0; i < indices.count; i++) {
        const Node* selector = indices.nodes[i];
        shd_enter_composite_type(datatype, s, selector);
    }
}

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count) {
    if (outputs_count > 1) {
        LARRAY(const Node*, extracted, outputs_count);
        for (size_t i = 0; i < outputs_count; i++)
            extracted[i] = extract_helper(a, value, shd_int32_literal(a, i));
        return shd_nodes(a, outputs_count, extracted);
    } else if (outputs_count == 1)
        return shd_singleton(value);
    else
        return shd_empty(a);
}

typedef struct {
    const Node* n;
    int i;
} Node_and_i;

static Node_and_i find_annotation_by_index(Nodes annotations, const char* name, int i) {
    for (size_t j = 0; j < annotations.count; j++) {
        const Node* an = annotations.nodes[j];
        if (strcmp(get_annotation_name(an), name) == 0) {
            assert(an->tag == AnnotationValues_TAG);
            Nodes an_values = an->payload.annotation_values.values;
            assert(an_values.count >= 2);
            int idx = shd_get_int_value(an_values.nodes[0], false);
            if (idx == i) {
                return (Node_and_i) { an_values.nodes[1], i };
            }
        }
    }
    return (Node_and_i) { NULL, -1 };
}

Type* shd_struct_type(IrArena* a, StructType payload) {
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = a,
        .type = NULL,
        .tag = StructType_TAG,
        .payload.struct_type = payload
    };
    Node* decl = _shd_create_node_helper(a, node, NULL);
    return decl;
}

String shd_get_struct_type_field_name(const Type* t, size_t i) {
    assert(t->tag == StructType_TAG);
    StructType r = t->payload.struct_type;
    assert(i < r.members.count);

    Node_and_i nj = find_annotation_by_index(t->annotations, "MemberName", i);
    if (nj.i >= 0)
        return shd_get_string_literal(t->arena, nj.n);

    return shd_fmt_string_irarena(t->arena, "_%d", i);
}

void shd_set_struct_type_field_name(Type* t, size_t i, String name) {
    IrArena* a = t->arena;
    Node_and_i nj = find_annotation_by_index(t->annotations, "MemberName", i);
    if (nj.i >= 0) {
        t->annotations = shd_remove_annotation(a, t->annotations, nj.i);
    }
    const Node* annotation = annotation_values_helper(a, "MemberName", mk_nodes(a, shd_uint32_literal(a, i), string_lit_helper(a, name)));
    shd_add_annotation(t, annotation);
}

const Node* shd_struct_type_with_members(IrArena* a, ShdStructFlags flags, Nodes member_types) {
    Node* struct_t = struct_type_helper(a, flags);
    shd_struct_type_set_members(struct_t, member_types);
    return struct_t;
}

const Node* shd_struct_type_with_members_named(IrArena* a, ShdStructFlags flags, Nodes member_types, Strings member_names) {
    Node* struct_t = struct_type_helper(a, flags);
    shd_struct_type_set_members_named(struct_t, member_types, member_names);
    return struct_t;
}

// call this to finish two-part struct definitions start with struct_type(flag)
void shd_struct_type_set_members(Node* struct_t, Nodes member_types) {
    struct_t->payload.struct_type.members = member_types;
}

void shd_struct_type_set_members_named(Node* struct_t, Nodes member_types, Strings member_names) {
    shd_struct_type_set_members(struct_t, member_types);
    shd_remove_annotation_by_name(struct_t, "MemberName");
    for (size_t i = 0; i < member_names.count; i++) {
        String member_name = member_names.strings[i];
        if (member_name)
            shd_set_struct_type_field_name(struct_t, i, member_name);
    }
}
