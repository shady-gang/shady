#include "fold.h"
#include "type.h"

#include <assert.h>

const Node* resolve_known_vars(const Node* node, bool stop_at_values) {
    if (node->tag == Variable_TAG) {
        const Node* instr = node->payload.var.instruction;
        if (instr) {
            switch (instr->type->tag) {
                case RecordType_TAG: {
                    // TODO handle tuples
                    return node;
                }
                default: {
                    assert(node->payload.var.output == 0);
                    if (!stop_at_values || is_value(instr))
                        return resolve_known_vars(instr, stop_at_values);
                }
            }
        }
    }
    return node;
}

bool is_zero(const Node* node) {
    node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (node->payload.int_literal.value == 0)
            return true;
    }
    return false;
}
bool is_one(const Node* node) {
    node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (node->payload.int_literal.value == 1)
            return true;
    }
    return false;
}

const Node* fold_prim_op(IrArena* arena, const Node* node) {
    PrimOp prim_op = node->payload.prim_op;
    switch (prim_op.op) {
        case add_op: {
            // If either operand is zero, destroy the add
            for (size_t i = 0; i < 2; i++)
                if (is_zero(prim_op.operands.nodes[i]))
                    return prim_op.operands.nodes[1 - i];
            break;
        }
        case mul_op: {
            for (size_t i = 0; i < 2; i++)
                if (is_zero(prim_op.operands.nodes[i]))
                    return int_literal(arena, (IntLiteral) {
                        .value = 0
                    });

            for (size_t i = 0; i < 2; i++)
                if (is_one(prim_op.operands.nodes[i]))
                    return prim_op.operands.nodes[1 - i];

            break;
        }
        case reinterpret_op:
        case convert_op:
            // get rid of identity casts
            if (is_subtype(prim_op.operands.nodes[0], without_qualifier(prim_op.operands.nodes[1]->type)))
                return prim_op.operands.nodes[1];
            break;
        default: break;
    }
    return node;
}

const Node* fold_block(IrArena* arena, const Node* node) {
    return node;
}

const Node* fold_node(IrArena* arena, const Node* instruction) {
    switch (instruction->tag) {
        case PrimOp_TAG: return fold_prim_op(arena, instruction);
        case Block_TAG: return fold_block(arena, instruction);
        default: return instruction;
    }
}
