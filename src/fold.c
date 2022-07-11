#include "block_builder.h"

#include <assert.h>

bool is_zero(const Node* node) {
    if (node->tag == IntLiteral_TAG) {
        if (node->payload.int_literal.value == 0)
            return true;
    }
    return false;
}

bool fold_primop(BlockBuilder* builder, PrimOp op, Nodes* folded_to) {
    switch (op.op) {
        case add_op: {
            // If either operand is zero, destroy the add
            for (size_t i = 0; i < 2; i++)
                if (is_zero(op.operands.nodes[i])) {
                    *folded_to = nodes(builder->arena, 1, (const Node* []) { op.operands.nodes[1 - i] });
                    return true;
                }
        }
        default: break;
    }
    return false;
}

bool fold_instruction(BlockBuilder* builder, const Node* instruction, Nodes* folded_to) {
    assert(instruction->tag != Let_TAG && "block_builder.c should unwrap those already");
    if (instruction->tag == PrimOp_TAG)
        return fold_primop(builder, instruction->payload.prim_op, folded_to);
    return false;
}
