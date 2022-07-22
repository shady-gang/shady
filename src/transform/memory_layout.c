#include "memory_layout.h"
#include "ir_gen_helpers.h"

#include "../log.h"
#include "../block_builder.h"

TypeMemLayout get_mem_layout(const CompilerConfig* config, IrArena* arena, const Type* type) {
    switch (type->tag) {
        case FnType_TAG:  error("Functions have an opaque memory representation");
        case PtrType_TAG: switch (type->payload.ptr_type.address_space) {
            case AsProgramCode: return get_mem_layout(config, arena, int32_type(arena));
            default: error("unhandled")
        }
        // case MaskType_TAG: return get_mem_layout(config, arena, int_type(arena));
        case Int_TAG:     return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
            .size_in_cells = 1,
        };
        case Float_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
            .size_in_cells = 1,
        };
        case Bool_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
            .size_in_cells = 1,
        };
        case QualifiedType_TAG: return get_mem_layout(config, arena, type->payload.qualified_type.type);
        case RecordType_TAG: error("TODO");
        default: error("not a known type");
    }
}

const Node* gen_deserialisation(BlockBuilder* instructions, const Type* element_type, const Node* arr, const Node* base_offset) {
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions->arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* value = gen_load(instructions, logical_ptr);
            const Node* zero = int_literal(instructions->arena, (IntLiteral) { .value_i8 = 0, .width = IntTy8 });
            return gen_primop(instructions, (PrimOp) {
                .op = neq_op,
                .operands = nodes(instructions->arena, 2, (const Node*[]) {value, zero})
            }).nodes[0];
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto ser_int;
            default: error("TODO")
        }
        // case MaskType_TAG:
        case Int_TAG: ser_int: {
            // TODO handle the cases where int size != arr element_t
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions->arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* value = gen_load(instructions, logical_ptr);
            // note: folding gets rid of identity casts
            value = gen_primop(instructions, (PrimOp) {.op = reinterpret_op, .operands = nodes(instructions->arena, 2, (const Node* []){ element_type, value})}).nodes[0];
            return value;
        }
        default: error("TODO");
    }
}

void gen_serialisation(BlockBuilder* instructions, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    switch (element_type->tag) {
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions->arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* zero = int_literal(instructions->arena, (IntLiteral) { .value_i8 = 0, .width = IntTy8 });
            const Node* one = int_literal(instructions->arena, (IntLiteral) { .value_i8 = 1, .width = IntTy8 });
            const Node* int_value = gen_primop(instructions, (PrimOp) {
                .op = select_op,
                .operands = nodes(instructions->arena, 3, (const Node*[]) { value, zero, one })
            }).nodes[0];
            gen_store(instructions, logical_ptr, int_value);
            return;
        }
        case PtrType_TAG: switch (element_type->payload.ptr_type.address_space) {
            case AsProgramCode: goto des_int;
            default: error("TODO")
        }
        // case MaskType_TAG:
        case Int_TAG: des_int: {
            // note: folding gets rid of identity casts
            // value = gen_primop(instructions, (PrimOp) {.op = reinterpret_op, .operands = nodes(instructions->arena, 2, (const Node* []){ int_type(instructions->arena), value})}).nodes[0];
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions->arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            gen_store(instructions, logical_ptr, value);
            return;
        }
        default: error("TODO");
    }
}
