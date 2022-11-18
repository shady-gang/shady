#include "memory_layout.h"
#include "ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

size_t bytes_to_i32_cells(size_t size_in_bytes) {
    assert(size_in_bytes % 4 == 0);
    return (size_in_bytes + 3) / 4;
}

typedef struct {
    TypeMemLayout mem_layout;
    size_t offset_in_bytes;
} FieldLayout;

static size_t get_record_layout(const CompilerConfig* config, IrArena* arena, const Node* record_type, FieldLayout* fields) {
    assert(record_type->tag == RecordType_TAG);
    size_t total_size = 0;
    Nodes member_types = record_type->payload.record_type.members;
    for (size_t i = 0; i < member_types.count; i++) {
        TypeMemLayout member_layout = get_mem_layout(config, arena, member_types.nodes[i]);
        if (fields) {
            fields[i].mem_layout = member_layout;
            fields[i].offset_in_bytes = total_size;
        }
        // TODO implement alignment rules ?
        total_size += member_layout.size_in_bytes;
    }
    return total_size;
}

size_t get_record_field_offset_in_bytes(const CompilerConfig* c, IrArena* a, const Type* t, size_t i) {
    assert(t->tag == RecordType_TAG);
    Nodes member_types = t->payload.record_type.members;
    assert(i < member_types.count);
    LARRAY(FieldLayout, fields, member_types.count);
    get_record_layout(c, a, t, fields);
    return fields[i].offset_in_bytes;
}

TypeMemLayout get_mem_layout(const CompilerConfig* config, IrArena* arena, const Type* type) {
    assert(is_type(type));
    switch (type->tag) {
        case FnType_TAG:  error("Functions have an opaque memory representation");
        case PtrType_TAG: switch (type->payload.ptr_type.address_space) {
            case AsProgramCode: return get_mem_layout(config, arena, int32_type(arena));
            default: error("unhandled")
        }
        // case MaskType_TAG: return get_mem_layout(config, arena, int_type(arena));
        case Int_TAG:     return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = type->payload.int_type.width == IntTy64 ? 8 : 4,
        };
        case Float_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
        };
        case Bool_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
        };
        case ArrType_TAG: {
            const Node* size = type->payload.arr_type.size;
            assert(size && "We can't know the full layout of arrays of unknown size !");
            size_t actual_size = extract_int_literal_value(size, false);
            TypeMemLayout element_layout = get_mem_layout(config, arena, type->payload.arr_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = actual_size * element_layout.size_in_bytes
            };
        }
        case QualifiedType_TAG: return get_mem_layout(config, arena, type->payload.qualified_type.type);
        case TypeDeclRef_TAG: return get_mem_layout(config, arena, type->payload.type_decl_ref.decl->payload.nom_type.body);
        case RecordType_TAG: {
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = get_record_layout(config, arena, type, NULL),
            };
        }
        default: error("not a known type");
    }
}

const Node* gen_deserialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset) {
    const Node* zero = int32_literal(bb->arena, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* value = gen_load(bb, logical_ptr);
            return gen_primop_ce(bb, neq_op, 2, (const Node*[]) {value, zero});
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto ser_int;
            default: error("TODO")
        }
        case Int_TAG: ser_int: {
            if (element_type->payload.int_type.width != IntTy64) {
                // One load suffices
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* value = gen_load(bb, logical_ptr);
                // cast into the appropriate width and throw other bits away
                // note: folding gets rid of identity casts
                value = gen_primop_e(bb, reinterpret_op, singleton(element_type), singleton(value));
                return value;
            } else {
                // We need to decompose this into two loads, then we use the merge routine
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
                const Node* lo = gen_load(bb, logical_ptr);
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, int32_literal(bb->arena, 1) });
                            logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset });
                const Node* hi = gen_load(bb, logical_ptr);
                return gen_merge_i32s_i64(bb, lo, hi);
            }
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, element_type, fields);
            LARRAY(const Node*, loaded, member_types.count);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, int32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
                loaded[i] = gen_deserialisation(config, bb, member_types.nodes[i], arr, field_offset);
                TypeMemLayout member_layout = get_mem_layout(config, bb->arena, member_types.nodes[i]);
            }
            return tuple(bb->arena, nodes(bb->arena, member_types.count, loaded));
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            const Node* body = gen_deserialisation(config, bb, nom->payload.nom_type.body, arr, base_offset);
            return first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = make_op, .type_arguments = singleton(element_type), .operands = singleton(body) })));
        }
        default: error("TODO");
    }
}

void gen_serialisation(const CompilerConfig* config, BodyBuilder* bb, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    const Node* zero = int32_literal(bb->arena, 0);
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset });
            const Node* one = int32_literal(bb->arena, 1);
            const Node* int_value = gen_primop_ce(bb, select_op, 3, (const Node*[]) { value, one, zero });
            gen_store(bb, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto des_int;
            default: error("TODO")
        }
        case Int_TAG: des_int: {
            // Same story as for deser
            if (element_type->payload.int_type.width != IntTy64) {
                value = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(value));
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                gen_store(bb, logical_ptr, value);
            } else {
                const Node* lo = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(value));
                const Node* hi = gen_primop_ce(bb, lshift_op, 2, (const Node* []){ value, int64_literal(bb->arena, 32) });
                hi = gen_primop_e(bb, reinterpret_op, singleton(int32_type(bb->arena)), singleton(hi));
                // TODO: make this dependant on the emulation array type
                const Node* logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, base_offset});
                gen_store(bb, logical_ptr, lo);
                const Node* hi_destination_offset = gen_primop_ce(bb, add_op, 2, (const Node* []) { base_offset, int32_literal(bb->arena, 1) });
                            logical_ptr = gen_primop_ce(bb, lea_op, 3, (const Node* []) { arr, zero, hi_destination_offset});
                gen_store(bb, logical_ptr, hi);
            }
            return;
        }
        case RecordType_TAG: {
            Nodes member_types = element_type->payload.record_type.members;
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(config, bb->arena, element_type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                const Node* extracted_value = first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = extract_op, .operands = mk_nodes(bb->arena, value, int32_literal(bb->arena, i)), .type_arguments = empty(bb->arena) })));
                const Node* field_offset = gen_primop_e(bb, add_op, empty(bb->arena), mk_nodes(bb->arena, base_offset, int32_literal(bb->arena, bytes_to_i32_cells(fields[i].offset_in_bytes))));
                gen_serialisation(config, bb, member_types.nodes[i], arr, field_offset, extracted_value);
            }
            return;
        }
        case TypeDeclRef_TAG: {
            const Node* nom = element_type->payload.type_decl_ref.decl;
            assert(nom && nom->tag == NominalType_TAG);
            const Node* extracted_value = first(bind_instruction(bb, prim_op(bb->arena, (PrimOp) { .op = extract_op, .operands = mk_nodes(bb->arena, value, int32_literal(bb->arena, 0)), .type_arguments = empty(bb->arena) })));
            gen_serialisation(config, bb, nom->payload.nom_type.body, arr, base_offset, extracted_value);
            return;
        }
        default: error("TODO");
    }
}
