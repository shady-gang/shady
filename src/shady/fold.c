#include "fold.h"

#include "log.h"
#include "list.h"

#include "type.h"
#include "portability.h"
#include "rewrite.h"

#include <assert.h>
#include <math.h>

static const Node* quote_single(IrArena* a, const Node* value) {
    return quote_helper(a, singleton(value));
}

static bool is_zero(const Node* node) {
    const IntLiteral* lit = resolve_to_int_literal(node);
    if (lit && get_int_literal_value(*lit, false) == 0)
        return true;
    return false;
}

static bool is_one(const Node* node) {
    const IntLiteral* lit = resolve_to_int_literal(node);
    if (lit && get_int_literal_value(*lit, false) == 1)
        return true;
    return false;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    PrimOp payload = node->payload.prim_op;

    LARRAY(const FloatLiteral*, float_literals, payload.operands.count);
    FloatSizes float_width;
    bool all_float_literals = true;

    LARRAY(const IntLiteral*, int_literals, payload.operands.count);
    bool all_int_literals = true;
    IntSizes int_width;
    bool is_signed;
    for (size_t i = 0; i < payload.operands.count; i++) {
        int_literals[i] = resolve_to_int_literal(payload.operands.nodes[i]);
        all_int_literals &= int_literals[i] != NULL;
        if (int_literals[i]) {
            int_width = int_literals[i]->width;
            is_signed = int_literals[i]->is_signed;
        }

        float_literals[i] = resolve_to_float_literal(payload.operands.nodes[i]);
        if (float_literals[i])
            float_width = float_literals[i]->width;
        all_float_literals &= float_literals[i] != NULL;
    }

#define UN_OP(primop, op) case primop##_op: \
if (all_int_literals)        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = int_width, .value = op int_literals[0]->value})); \
else if (all_float_literals) return quote_single(arena, fp_literal_helper(arena, float_width, op get_float_literal_value(*float_literals[0]))); \
else break;

#define BIN_OP(primop, op) case primop##_op: \
if (all_int_literals)        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = int_width, .value = int_literals[0]->value op int_literals[1]->value })); \
else if (all_float_literals) return quote_single(arena, fp_literal_helper(arena, float_width, get_float_literal_value(*float_literals[0]) op get_float_literal_value(*float_literals[1]))); \
break;

    if (all_int_literals || all_float_literals) {
        switch (payload.op) {
            UN_OP(neg, -)
            BIN_OP(add, +)
            BIN_OP(sub, -)
            BIN_OP(mul, *)
            BIN_OP(div, /)
            case mod_op:
                if (all_int_literals)
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = int_width, .value = int_literals[0]->value % int_literals[1]->value }));
                else
                    return quote_single(arena, fp_literal_helper(arena, float_width, fmod(get_float_literal_value(*float_literals[0]), get_float_literal_value(*float_literals[1]))));
            case reinterpret_op: {
                const Type* dst_t = first(payload.type_arguments);
                uint64_t raw_value = int_literals[0] ? int_literals[0]->value : float_literals[0]->value;
                if (dst_t->tag == Int_TAG) {
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = raw_value }));
                } else if (dst_t->tag == Float_TAG) {
                    return quote_single(arena, float_literal(arena, (FloatLiteral) { .width = dst_t->payload.float_type.width, .value = raw_value }));
                }
                break;
            }
            case convert_op: {
                const Type* dst_t = first(payload.type_arguments);
                uint64_t bitmask = 0;
                if (get_type_bitwidth(dst_t) == 64)
                    bitmask = UINT64_MAX;
                else
                    bitmask = ~(UINT64_MAX << get_type_bitwidth(dst_t));
                if (dst_t->tag == Int_TAG) {
                    if (all_int_literals) {
                        uint64_t old_value = get_int_literal_value(*int_literals[0], int_literals[0]->is_signed);
                        uint64_t value = old_value & bitmask;
                        return quote_single(arena, int_literal(arena, (IntLiteral) {.is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value}));
                    } else if (all_float_literals) {
                        double old_value = get_float_literal_value(*float_literals[0]);
                        int64_t value = old_value;
                        return quote_single(arena, int_literal(arena, (IntLiteral) {.is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value}));
                    }
                } else if (dst_t->tag == Float_TAG) {
                    if (all_int_literals) {
                        uint64_t old_value = get_int_literal_value(*int_literals[0], int_literals[0]->is_signed);
                        double value = old_value;
                        return quote_single(arena, fp_literal_helper(arena, dst_t->payload.float_type.width, value));
                    } else if (all_float_literals) {
                        double old_value = get_float_literal_value(*float_literals[0]);
                        return quote_single(arena, float_literal(arena, (FloatLiteral) { .width = dst_t->payload.float_type.width, .value = old_value }));
                    }
                }
                break;
            }
            default: break;
        }
    }

    switch (payload.op) {
        case add_op: {
            // If either operand is zero, destroy the add
            for (size_t i = 0; i < 2; i++)
                if (is_zero(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[1 - i]);
            break;
        }
        case sub_op: {
            // If second operand is zero, return the first one
            if (is_zero(payload.operands.nodes[1]))
                return quote_single(arena, payload.operands.nodes[0]);
            // if first operand is zero, invert the second one
            if (is_zero(payload.operands.nodes[0]))
                return prim_op(arena, (PrimOp) { .op = neg_op, .operands = singleton(payload.operands.nodes[1]), .type_arguments = empty(arena) });
            break;
        }
        case mul_op: {
            for (size_t i = 0; i < 2; i++)
                if (is_zero(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[i]); // return zero !

            for (size_t i = 0; i < 2; i++)
                if (is_one(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[1 - i]);

            break;
        }
        case div_op: {
            // If second operand is one, return the first one
            if (is_one(payload.operands.nodes[1]))
                return quote_single(arena, payload.operands.nodes[0]);
            break;
        }
        case subgroup_broadcast_first_op: {
            const Node* value = first(payload.operands);
            if (is_qualified_type_uniform(value->type))
                return quote_single(arena, value);
            break;
        }
        case store_op: {
            if (first(payload.operands)->tag == Undef_TAG) {
                return quote_helper(arena, empty(arena));
            }
            break;
        }
        case load_op: {
            if (first(payload.operands)->tag == Undef_TAG) {
                return quote_single(arena, undef(arena, (Undef) { .type = get_unqualified_type(node->type) }));
            }
            break;
        }
        case reinterpret_op:
        case convert_op:
            if (first(payload.operands)->tag == Undef_TAG) {
                return quote_single(arena, undef(arena, (Undef) { .type = get_unqualified_type(node->type) }));
            }
            // get rid of identity casts
            if (payload.type_arguments.nodes[0] == get_unqualified_type(payload.operands.nodes[0]->type))
                return quote_single(arena, payload.operands.nodes[0]);
            break;
        case lea_op:
            if (first(payload.operands)->tag == Undef_TAG) {
                return quote_single(arena, undef(arena, (Undef) { .type = get_unqualified_type(node->type) }));
            }
            break;
        default: break;
    }
    return node;
}

static bool is_unreachable_case(const Node* c) {
    assert(c && c->tag == Case_TAG);
    const Node* b = get_abstraction_body(c);
    return b->tag == Unreachable_TAG;
}

static bool is_unreachable_jump(const Node* j) {
    assert(j && j->tag == Jump_TAG);
    const Node* dst = j->payload.jump.target;
    assert(dst->tag == BasicBlock_TAG);
    const Node* b = get_abstraction_body(dst);
    return b->tag == Unreachable_TAG;
}

const Node* fold_node(IrArena* arena, const Node* node) {
    const Node* folded = node;
    switch (node->tag) {
        case PrimOp_TAG: folded = fold_prim_op(arena, node); break;
        case Body_TAG: {
            Nodes instructions = node->payload.body.instructions;
            // Body([], T) => T
            if (instructions.count == 0)
                return node->payload.body.terminator;
            const Node* terminator = node->payload.body.terminator;
            // Body(I, Body(I2, T)) => Body(I ++ I2, T)
            if (terminator->tag == Body_TAG) {
                return body(arena, (Body) {
                    .instructions = concat_nodes(arena, instructions, terminator->payload.body.instructions),
                    .terminator = terminator->payload.body.terminator
                });
            }
            // Body(I ++ InsertHelper(..., Yield())
            for (size_t i = 0; i < instructions.count; i++) {
                const Node* instruction = instructions.nodes[i];
                if (instruction->tag == InsertHelper_TAG) {
                    const Node* tail = body(arena, (Body) {
                        .instructions = nodes(arena, instructions.count - i - 1, &instructions.nodes[i + 1]),
                        .terminator = node->payload.body.terminator
                    });
                    error("TODO");
                }
            }
            break;
        }
        case Branch_TAG: {
            Branch payload = node->payload.branch;
            if (arena->config.optimisations.delete_unreachable_structured_cases) {
                if (is_unreachable_jump(payload.false_destination))
                    return payload.true_destination;
                else if (is_unreachable_jump(payload.true_destination))
                    return payload.false_destination;
            }
            break;
        }
        case Switch_TAG: {
            if (!arena->config.optimisations.delete_unreachable_structured_cases)
                break;
            Switch payload = node->payload.br_switch;
            Nodes old_destinations = payload.destinations;
            LARRAY(const Node*, literals, old_destinations.count);
            LARRAY(const Node*, destinations, old_destinations.count);
            size_t new_cases_count = 0;
            for (size_t i = 0; i < old_destinations.count; i++) {
                const Node* j = old_destinations.nodes[i];
                if (is_unreachable_jump(j))
                    continue;
                literals[new_cases_count] = payload.literals.nodes[i];
                destinations[new_cases_count] = node->payload.structured_match.cases.nodes[i];
                new_cases_count++;
            }
            if (new_cases_count == old_destinations.count)
                break;

            if (new_cases_count == 1 && is_unreachable_jump(payload.default_destination))
                return destinations[0];

            if (new_cases_count == 0)
                return payload.default_destination;

            return br_switch(arena, (Switch) {
                .inspect = payload.inspect,
                .default_destination = payload.default_destination,
                .literals = nodes(arena, new_cases_count, literals),
                .destinations = nodes(arena, new_cases_count, destinations),
            });
        }
        default: break;
    }

    // catch bad folding rules that mess things up
    if (is_value(node)) assert(is_value(folded));
    if (is_instruction(node)) assert(is_instruction(folded));
    if (is_terminator(node)) assert(is_terminator(folded));

    if (node->type)
        assert(is_subtype(node->type, folded->type));

    return folded;
}
