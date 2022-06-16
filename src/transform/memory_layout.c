#include "memory_layout.h"
#include "ir_gen_helpers.h"

#include "../log.h"

TypeMemLayout get_mem_layout(const CompilerConfig* config, const Type* type) {
    switch (type->tag) {
        case FnType_TAG:  error("Functions have an opaque memory representation");
        case PtrType_TAG: error("TODO");
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
        case QualifiedType_TAG: return get_mem_layout(config, type->payload.qualified_type.type);
        case RecordType_TAG: error("TODO");
        default: error("not a known type");
    }
}

const Node* gen_deserialisation(Instructions instructions, const Type* element_type, const Node* arr, const Node* base_offset) {
    switch (element_type->tag) {
        case Int_TAG: {
            // TODO handle the cases where int size != arr element_t
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions.arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* value = gen_load(instructions, logical_ptr);
            return value;
        }
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions.arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* value = gen_load(instructions, logical_ptr);
            const Node* zero = int_literal(instructions.arena, (IntLiteral) { .value = 0 });
            return gen_primop(instructions, (PrimOp) {
                .op = neq_op,
                .operands = nodes(instructions.arena, 2, (const Node*[]) {value, zero})
            }).nodes[0];
        }
        default: error("TODO");
    }
}

void gen_serialisation(Instructions instructions, const Type* element_type, const Node* arr, const Node* base_offset, const Node* value) {
    switch (element_type->tag) {
        case Int_TAG: {
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions.arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            gen_store(instructions, logical_ptr, value);
            return;
        }
        case Bool_TAG: {
            const Node* logical_ptr = gen_primop(instructions, (PrimOp) {
                .op = lea_op,
                .operands = nodes(instructions.arena, 3, (const Node* []) { arr, NULL, base_offset})
            }).nodes[0];
            const Node* zero = int_literal(instructions.arena, (IntLiteral) { .value = 0 });
            const Node* one = int_literal(instructions.arena, (IntLiteral) { .value = 1 });
            const Node* int_value = gen_primop(instructions, (PrimOp) {
                .op = select_op,
                .operands = nodes(instructions.arena, 3, (const Node*[]) { value, zero, one })
            }).nodes[0];
            gen_store(instructions, logical_ptr, int_value);
            return;
        }
        default: error("TODO");
    }
}
