#include "shady/ir/memory_layout.h"

#include "ir_gen_helpers.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

inline static size_t round_up(size_t a, size_t b) {
    if (b == 0)
        return a;
    size_t divided = (a + b - 1) / b;
    return divided * b;
}

static int maxof(int a, int b) {
    if (a > b)
        return a;
    return b;
}

TypeMemLayout shd_get_record_layout(IrArena* a, const Node* record_type, FieldLayout* fields) {
    assert(record_type->tag == RecordType_TAG);

    size_t offset = 0;
    size_t max_align = 0;

    Nodes member_types = record_type->payload.record_type.members;
    for (size_t i = 0; i < member_types.count; i++) {
        TypeMemLayout member_layout = shd_get_mem_layout(a, member_types.nodes[i]);
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

size_t shd_get_record_field_offset_in_bytes(IrArena* a, const Type* t, size_t i) {
    assert(t->tag == RecordType_TAG);
    Nodes member_types = t->payload.record_type.members;
    assert(i < member_types.count);
    LARRAY(FieldLayout, fields, member_types.count);
    shd_get_record_layout(a, t, fields);
    return fields[i].offset_in_bytes;
}

TypeMemLayout shd_get_mem_layout(IrArena* a, const Type* type) {
    size_t base_word_size = int_size_in_bytes(shd_get_arena_config(a)->memory.word_size);
    assert(is_type(type));
    switch (type->tag) {
        case FnType_TAG:  shd_error("Functions have an opaque memory representation");
        case PtrType_TAG: switch (type->payload.ptr_type.address_space) {
            case AsPrivate:
            case AsSubgroup:
            case AsShared:
            case AsGlobal:
            case AsGeneric: return shd_get_mem_layout(a, int_type(a, (Int) { .width = shd_get_arena_config(a)->memory.ptr_size, .is_signed = false })); // TODO: use per-as layout
            default: shd_error("Pointers in address space '%s' does not have a defined memory layout", get_address_space_name(type->payload.ptr_type.address_space));
        }
        case Int_TAG:     return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = int_size_in_bytes(type->payload.int_type.width),
            .alignment_in_bytes = maxof(int_size_in_bytes(type->payload.int_type.width), base_word_size),
        };
        case Float_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = float_size_in_bytes(type->payload.float_type.width),
            .alignment_in_bytes = maxof(float_size_in_bytes(type->payload.float_type.width), base_word_size),
        };
        case Bool_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = base_word_size,
            .alignment_in_bytes = base_word_size,
        };
        case ArrType_TAG: {
            const Node* size = type->payload.arr_type.size;
            assert(size && "We can't know the full layout of arrays of unknown size !");
            size_t actual_size = shd_get_int_literal_value(*shd_resolve_to_int_literal(size), false);
            TypeMemLayout element_layout = shd_get_mem_layout(a, type->payload.arr_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = actual_size * element_layout.size_in_bytes,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case PackType_TAG: {
            size_t width = type->payload.pack_type.width;
            TypeMemLayout element_layout = shd_get_mem_layout(a, type->payload.pack_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = width * element_layout.size_in_bytes /* TODO Vulkan vec3 -> vec4 alignment rules ? */,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case QualifiedType_TAG: return shd_get_mem_layout(a, type->payload.qualified_type.type);
        case TypeDeclRef_TAG: return shd_get_mem_layout(a, type->payload.type_decl_ref.decl->payload.nom_type.body);
        case RecordType_TAG: return shd_get_record_layout(a, type, NULL);
        default: shd_error("not a known type");
    }
}

const Node* shd_bytes_to_words(BodyBuilder* bb, const Node* bytes) {
    IrArena* a = bytes->arena;
    const Type* word_type = int_type(a, (Int) { .width = shd_get_arena_config(a)->memory.word_size, .is_signed = false });
    size_t word_width = shd_get_type_bitwidth(word_type);
    const Node* bytes_per_word = size_t_literal(a, word_width / 8);
    return gen_primop_e(bb, div_op, shd_empty(a), mk_nodes(a, bytes, bytes_per_word));
}

uint64_t shd_bytes_to_words_static(const IrArena* a, uint64_t bytes) {
    uint64_t word_width = int_size_in_bytes(shd_get_arena_config(a)->memory.word_size);
    return bytes / word_width;
}

IntSizes shd_float_to_int_width(FloatSizes width) {
    switch (width) {
        case FloatTy16: return IntTy16;
        case FloatTy32: return IntTy32;
        case FloatTy64: return IntTy64;
    }
}
