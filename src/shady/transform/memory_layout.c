#include "memory_layout.h"
#include "ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

#include "../type.h"

#include <assert.h>

inline static size_t round_up(size_t a, size_t b) {
    size_t divided = (a + b - 1) / b;
    return divided * b;
}

TypeMemLayout get_record_layout(IrArena* a, const Node* record_type, FieldLayout* fields) {
    assert(record_type->tag == RecordType_TAG);

    size_t offset = 0;
    size_t max_align = 0;

    Nodes member_types = record_type->payload.record_type.members;
    for (size_t i = 0; i < member_types.count; i++) {
        TypeMemLayout member_layout = get_mem_layout(a, member_types.nodes[i]);
        offset = round_up(offset, member_layout.alignment_in_bytes);
        if (fields) {
            fields[i].mem_layout = member_layout;
            fields[i].offset_in_bytes = offset;
        }
        offset += member_layout.size_in_bytes;
        if (member_layout.alignment_in_bytes > max_align)
            max_align = member_layout.alignment_in_bytes;
    }

    return (TypeMemLayout) {
        .type = record_type,
        .size_in_bytes = round_up(offset, max_align),
        .alignment_in_bytes = max_align,
    };
}

size_t get_record_field_offset_in_bytes(IrArena* a, const Type* t, size_t i) {
    assert(t->tag == RecordType_TAG);
    Nodes member_types = t->payload.record_type.members;
    assert(i < member_types.count);
    LARRAY(FieldLayout, fields, member_types.count);
    get_record_layout(a, t, fields);
    return fields[i].offset_in_bytes;
}

TypeMemLayout get_mem_layout(IrArena* a, const Type* type) {
    assert(is_type(type));
    switch (type->tag) {
        case FnType_TAG:  error("Functions have an opaque memory representation");
        case PtrType_TAG: switch (type->payload.ptr_type.address_space) {
            case AsProgramCode:
            case AsPrivatePhysical:
            case AsSubgroupPhysical:
            case AsSharedPhysical:
            case AsGlobalPhysical:
            case AsGeneric: return get_mem_layout(a, int_type(a, (Int) { .width = a->config.memory.ptr_size, .is_signed = false }));
            default: error_print("as: %d", type->payload.ptr_type.address_space); error("unhandled address space")
        }
        case Int_TAG:     return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = get_type_bitwidth(type) / 8,
            .alignment_in_bytes = type->payload.int_type.width == IntTy64 ? 8 : 4,
        };
        case Float_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = get_type_bitwidth(type),
            .alignment_in_bytes = type->payload.float_type.width == FloatTy64 ? 8 : 4,
        };
        case Bool_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
            .alignment_in_bytes = 4,
        };
        case ArrType_TAG: {
            const Node* size = type->payload.arr_type.size;
            assert(size && "We can't know the full layout of arrays of unknown size !");
            size_t actual_size = get_int_literal_value(size, false);
            TypeMemLayout element_layout = get_mem_layout(a, type->payload.arr_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = actual_size * element_layout.size_in_bytes,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case PackType_TAG: {
            size_t width = type->payload.pack_type.width;
            TypeMemLayout element_layout = get_mem_layout(a, type->payload.pack_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = width * element_layout.size_in_bytes /* TODO Vulkan vec3 -> vec4 alignment rules ? */,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case QualifiedType_TAG: return get_mem_layout(a, type->payload.qualified_type.type);
        case TypeDeclRef_TAG: return get_mem_layout(a, type->payload.type_decl_ref.decl->payload.nom_type.body);
        case RecordType_TAG: return get_record_layout(a, type, NULL);
        default: error("not a known type");
    }
}
