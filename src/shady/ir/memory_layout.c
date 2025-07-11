#include "shady/ir/memory_layout.h"
#include "shady/ir/float.h"
#include "shady/ir/type.h"

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
    assert(record_type->tag == StructType_TAG);

    size_t offset = 0;
    size_t max_align = 0;

    Nodes member_types = record_type->payload.struct_type.members;
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
    assert(t->tag == StructType_TAG);
    Nodes member_types = t->payload.struct_type.members;
    assert(i < member_types.count);
    LARRAY(FieldLayout, fields, member_types.count);
    shd_get_record_layout(a, t, fields);
    return fields[i].offset_in_bytes;
}

size_t shd_get_composite_index_offset_in_bytes(IrArena* a, const Type* t, size_t i) {
    switch (t->tag) {
        case StructType_TAG: return shd_get_record_field_offset_in_bytes(a, t, i);
        case ArrType_TAG: {
            TypeMemLayout element_layout = shd_get_mem_layout(a, t->payload.arr_type.element_type);
            assert(element_layout.size_in_bytes > 0);
            return element_layout.size_in_bytes * i;
        }
        case VectorType_TAG: {
            TypeMemLayout element_layout = shd_get_mem_layout(a, t->payload.vector_type.element_type);
            assert(element_layout.size_in_bytes > 0);
            return element_layout.size_in_bytes * i;
        }
        default: shd_error_die();
    }
}

TypeMemLayout shd_get_mem_layout(IrArena* a, const Type* type) {
    size_t base_word_size = int_size_in_bytes(shd_get_arena_config(a)->target.memory.word_size);
    assert(is_type(type));
    switch (type->tag) {
        case FnType_TAG:  shd_error("Functions have an opaque memory representation");
        case PtrType_TAG: switch (type->payload.ptr_type.address_space) {
            case AsFunction:
            case AsPrivate:
            case AsSubgroup:
            case AsShared:
            case AsGlobal:
            case AsGeneric: {
                size_t size_in_bytes = int_size_in_bytes(shd_get_arena_config(a)->target.memory.ptr_size);
                return (TypeMemLayout) {
                    .type = type,
                    .alignment_in_bytes = size_in_bytes,
                    .size_in_bytes = size_in_bytes,
                };
            }
            default: shd_error("Pointers in address space '%s' does not have a defined memory layout", shd_get_address_space_name(type->payload.ptr_type.address_space));
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
        case VectorType_TAG: {
            size_t width = type->payload.vector_type.width;
            TypeMemLayout element_layout = shd_get_mem_layout(a, type->payload.vector_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = width * element_layout.size_in_bytes /* TODO Vulkan vec3 -> vec4 alignment rules ? */,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case MatrixType_TAG: {
            size_t width = type->payload.matrix_type.columns;
            TypeMemLayout element_layout = shd_get_mem_layout(a, type->payload.matrix_type.element_type);
            return (TypeMemLayout) {
                .type = type,
                .size_in_bytes = width * element_layout.size_in_bytes /* TODO Vulkan vec3 -> vec4 alignment rules ? */,
                .alignment_in_bytes = element_layout.alignment_in_bytes
            };
        }
        case QualifiedType_TAG: return shd_get_mem_layout(a, type->payload.qualified_type.type);
        case StructType_TAG: return shd_get_record_layout(a, type, NULL);
        default: shd_error("not a known type");
    }
}

const Node* shd_bytes_to_words(BodyBuilder* bb, const Node* bytes) {
    IrArena* a = bytes->arena;
    const Type* word_type = int_type(a, (Int) { .width = shd_get_arena_config(a)->target.memory.word_size, .is_signed = false });
    size_t word_width = shd_get_type_bitwidth(word_type);
    const Type* bytes_t = shd_get_unqualified_type(bytes->type);
    assert(bytes_t->tag == Int_TAG);
    const Node* bytes_per_word = int_literal_helper(a, bytes_t->payload.int_type.width, bytes_t->payload.int_type.is_signed, word_width / 8);
    return prim_op_helper(a, div_op, mk_nodes(a, bytes, bytes_per_word));
}

uint64_t shd_bytes_to_words_static(const IrArena* a, uint64_t bytes) {
    uint64_t word_width = int_size_in_bytes(shd_get_arena_config(a)->target.memory.word_size);
    return bytes / word_width;
}

ShdIntSize shd_float_to_int_width(ShdFloatFormat width) {
    switch (width) {
        case ShdFloatFormat16: return ShdIntSize16;
        case ShdFloatFormat32: return ShdIntSize32;
        case ShdFloatFormat64: return ShdIntSize64;
    }
}

size_t shd_get_type_bitwidth(const Type* t) {
    const ArenaConfig* aconfig = shd_get_arena_config(t->arena);
    switch (t->tag) {
        case Int_TAG: return int_size_in_bytes(t->payload.int_type.width) * 8;
        case Float_TAG: return float_size_in_bytes(t->payload.float_type.width) * 8;
        case PtrType_TAG: {
            if (t->payload.ptr_type.address_space == AsCode)
                return int_size_in_bytes(aconfig->target.memory.fn_ptr_size) * 8;
            if (aconfig->target.memory.address_spaces[t->payload.ptr_type.address_space].physical)
                return int_size_in_bytes(aconfig->target.memory.ptr_size) * 8;
            break;
        }
        default: break;
    }
    return SIZE_MAX;
}

const Node* _shd_lea_helper(IrArena* a, const Node* ptr, const Node* offset, Nodes indices) {
    const Node* lea = ptr_array_element_offset(a, (PtrArrayElementOffset) {
        .ptr = ptr,
        .offset = offset,
    });
    for (size_t i = 0; i < indices.count; i++) {
        lea = ptr_composite_element(a, (PtrCompositeElement) {
            .ptr = lea,
            .index = indices.nodes[i],
        });
    }
    return lea;
}
