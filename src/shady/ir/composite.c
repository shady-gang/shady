#include "shady/ir/composite.h"

#include "ir_private.h"
#include "transform/ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

const Node* shd_extract_helper(IrArena* a, const Node* base, Nodes selectors) {
    LARRAY(const Node*, ops, 1 + selectors.count);
    ops[0] = base;
    for (size_t i = 0; i < selectors.count; i++)
        ops[1 + i] = selectors.nodes[i];
    return prim_op_helper(a, extract_op, shd_empty(a), shd_nodes(a, 1 + selectors.count, ops));
}

const Node* shd_extract_single_helper(IrArena* a, const Node* composite, const Node* index) {
    return prim_op_helper(a, extract_op, shd_empty(a), mk_nodes(a, composite, index));
}

const Node* shd_maybe_tuple_helper(IrArena* a, Nodes values) {
    if (values.count == 1)
        return shd_first(values);
    return shd_tuple_helper(a, values);
}

const Node* shd_tuple_helper(IrArena* a, Nodes contents) {
    const Type* t = NULL;
    if (a->config.check_types) {
        // infer the type of the tuple
        Nodes member_types = shd_get_values_types(a, contents);
        t = record_type(a, (RecordType) {.members = shd_strip_qualifiers(a, member_types)});
    }

    return composite_helper(a, t, contents);
}

void shd_enter_composite_type(const Type** datatype, bool* uniform, const Node* selector, bool allow_entering_pack) {
    const Type* current_type = *datatype;

    if (selector->arena->config.check_types) {
        const Type* selector_type = selector->type;
        bool selector_uniform = shd_deconstruct_qualified_type(&selector_type);
        assert(selector_type->tag == Int_TAG && "selectors must be integers");
        *uniform &= selector_uniform;
    }

    try_again:
    switch (current_type->tag) {
        case RecordType_TAG: {
            size_t selector_value = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
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
            size_t selector_value = shd_get_int_literal_value(*shd_resolve_to_int_literal(selector), false);
            assert(selector_value < current_type->payload.pack_type.width);
            current_type = current_type->payload.pack_type.element_type;
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

void shd_enter_composite_type_indices(const Type** datatype, bool* uniform, Nodes indices, bool allow_entering_pack) {
    for(size_t i = 0; i < indices.count; i++) {
        const Node* selector = indices.nodes[i];
        shd_enter_composite_type(datatype, uniform, selector, allow_entering_pack);
    }
}

Nodes shd_deconstruct_composite(IrArena* a, const Node* value, size_t outputs_count) {
    if (outputs_count > 1) {
        LARRAY(const Node*, extracted, outputs_count);
        for (size_t i = 0; i < outputs_count; i++)
            extracted[i] = shd_extract_single_helper(a, value, shd_int32_literal(a, i));
        return shd_nodes(a, outputs_count, extracted);
    } else if (outputs_count == 1)
        return shd_singleton(value);
    else
        return shd_empty(a);
}
