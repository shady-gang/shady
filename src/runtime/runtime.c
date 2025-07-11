#include "shady/runtime/runtime.h"

#include "shady/ir/grammar.h"
#include "shady/ir/type.h"
#include "shady/ir/memory_layout.h"

#include "portability.h"

void shd_rt_materialize_constant_at(unsigned char* tgt, const Node* value) {
    assert(tgt);
    IrArena* a = value->arena;
    switch (value->tag) {
        case IntLiteral_TAG: {
            switch (value->payload.int_literal.width) {
                case ShdIntSize8: *((uint8_t*) tgt) = (uint8_t) (value->payload.int_literal.value & 0xFF); break;
                case ShdIntSize16: *((uint16_t*) tgt) = (uint16_t) (value->payload.int_literal.value & 0xFFFF); break;
                case ShdIntSize32: *((uint32_t*) tgt) = (uint32_t) (value->payload.int_literal.value & 0xFFFFFFFF); break;
                case ShdIntSize64: *((uint64_t*) tgt) = (uint64_t) (value->payload.int_literal.value); break;
            }
            break;
        }
        case FloatLiteral_TAG: {
            switch (value->payload.float_literal.width) {
                case ShdFloatFormat16: *((uint16_t*) tgt) = (uint16_t) (value->payload.float_literal.value & 0xFFFF); break;
                case ShdFloatFormat32: *((uint32_t*) tgt) = (uint32_t) (value->payload.float_literal.value & 0xFFFFFFFF); break;
                case ShdFloatFormat64: *((uint64_t*) tgt) = (uint64_t) (value->payload.float_literal.value); break;
            }
            break;
        }
        case Composite_TAG: {
            Nodes values = value->payload.composite.contents;
            const Type* struct_t = value->payload.composite.type;

            if (struct_t->tag == StructType_TAG) {
                LARRAY(FieldLayout, fields, values.count);
                shd_get_record_layout(a, struct_t, fields);
                for (size_t i = 0; i < values.count; i++) {
                    // TypeMemLayout layout = get_mem_layout(value->arena, get_unqualified_type(element->type));
                    shd_rt_materialize_constant_at(tgt + fields[i].offset_in_bytes, values.nodes[i]);
                }
            } else if (struct_t->tag == ArrType_TAG) {
                for (size_t i = 0; i < values.count; i++) {
                    TypeMemLayout layout = shd_get_mem_layout(value->arena, shd_get_unqualified_type(values.nodes[i]->type));
                    shd_rt_materialize_constant_at(tgt, values.nodes[i]);
                    tgt += layout.size_in_bytes;
                }
            } else {
                assert(false);
            }
            break;
        }
        default:
            assert(false);
    }
}


void shd_rt_materialize_constant(const Node* value, size_t* size, void* data) {
    IrArena* a = value->arena;
    TypeMemLayout value_layout = shd_get_mem_layout(a, shd_get_unqualified_type(value->type));
    *size = value_layout.size_in_bytes;
    if (data)
        shd_rt_materialize_constant_at(data, value);
}
