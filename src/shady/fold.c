#include "fold.h"
#include "type.h"
#include "portability.h"
#include "rewrite.h"

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

static bool is_zero(const Node* node) {
    node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (extract_int_literal_value(node, false) == 0)
            return true;
    }
    return false;
}

static bool is_one(const Node* node) {
    node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (extract_int_literal_value(node, false) == 1)
            return true;
    }
    return false;
}

/// Substitutes the parameters for the arguments in the function body
static const Node* reduce_beta(const Node* fn, Nodes args) {
    assert(is_abstraction(fn));
    Nodes params = get_abstraction_params(fn);
    const Node* body = get_abstraction_body(fn);
    assert(body);

    assert(params.count == args.count);
    Rewriter r = create_substituter(get_abstraction_module(fn));
    for (size_t i = 0; i < args.count; i++)
        register_processed(&r, params.nodes[i], args.nodes[i]);
    const Node* specialized = rewrite_node(&r, body);
    assert(specialized);
    destroy_rewriter(&r);
    return specialized;
}

static const Node* fold_let(IrArena* arena, const Node* node) {
    assert(node->tag == Let_TAG);
    const Node* instruction = node->payload.let.instruction;
    const Node* tail = node->payload.let.tail;
    switch (instruction->tag) {
        case PrimOp_TAG: {
            if (instruction->payload.prim_op.op == quote_op) {
                if (tail->payload.anon_lam.body) {
                    return reduce_beta(tail, instruction->payload.prim_op.operands);
                }
            }
            break;
        }
        default: break;
    }

    return node;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    PrimOp prim_op = node->payload.prim_op;
    switch (prim_op.op) {
        case add_op: {
            // If either operand is zero, destroy the add
            for (size_t i = 0; i < 2; i++)
                if (is_zero(prim_op.operands.nodes[i]))
                    return quote_single(arena, prim_op.operands.nodes[1 - i]);
            break;
        }
        case mul_op: {
            for (size_t i = 0; i < 2; i++)
                if (is_zero(prim_op.operands.nodes[i]))
                    return quote_single(arena, prim_op.operands.nodes[i]); // return zero !

            for (size_t i = 0; i < 2; i++)
                if (is_one(prim_op.operands.nodes[i]))
                    return quote_single(arena, prim_op.operands.nodes[1 - i]);

            break;
        }
        case reinterpret_op:
        case convert_op:
            // get rid of identity casts
            if (is_subtype(prim_op.type_arguments.nodes[0], extract_operand_type(prim_op.operands.nodes[0]->type)))
                return quote_single(arena, prim_op.operands.nodes[0]);
            break;
        default: break;
    }
    return node;
}

const Node* fold_node(IrArena* arena, const Node* node) {
    const Node* folded = node;
    switch (node->tag) {
        case Let_TAG: folded = fold_let(arena, node); break;
        case PrimOp_TAG: folded = fold_prim_op(arena, node); break;
        default: break;
    }

    // catch bad folding rules that mess things up
    if (is_value(node)) assert(is_value(folded));
    if (is_instruction(node)) assert(is_instruction(folded));
    if (is_terminator(node)) assert(is_terminator(folded));

    return folded;
}
