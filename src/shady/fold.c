#include "fold.h"

#include "log.h"

#include "type.h"
#include "portability.h"
#include "rewrite.h"

#include <assert.h>

static const Node* quote_single(IrArena* a, const Node* value) {
    return quote_helper(a, singleton(value));
}

/*const Node* resolve_known_vars(const Node* node, bool stop_at_values) {
    if (node->tag == Variable_TAG) {
        const Node* abs = node->payload.var.abs;
        if (abs->tag == Case_TAG && abs->payload.case_.usage) {
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
    }
    return node;
}*/

static bool is_zero(const Node* node) {
    //node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (get_int_literal_value(node, false) == 0)
            return true;
    }
    return false;
}

static bool is_one(const Node* node) {
    //node = resolve_known_vars(node, false);
    if (node->tag == IntLiteral_TAG) {
        if (get_int_literal_value(node, false) == 1)
            return true;
    }
    return false;
}

static const Node* fold_let(IrArena* arena, const Node* node) {
    assert(node->tag == Let_TAG);
    const Node* instruction = node->payload.let.instruction;
    const Node* tail = node->payload.let.tail;
    switch (instruction->tag) {
        case PrimOp_TAG: {
            BodyBuilder* bb = begin_body(arena);
            Nodes operands = instruction->payload.prim_op.operands;
            LARRAY(const Node*, noperands, operands.count);
            bool modified = false;
            for (size_t i = 0; i < operands.count; i++) {
                noperands[i] = operands.nodes[i];
                if (operands.nodes[i]->tag == AntiQuote_TAG) {
                    noperands[i] = first(bind_instruction(bb, operands.nodes[i]->payload.anti_quote.instruction));
                    modified = true;
                }
            }
            if (!modified) {
                cancel_body(bb);
                break;
            }
            Nodes results = bind_instruction(bb, prim_op(arena, (PrimOp) {
                .op = instruction->payload.prim_op.op,
                .operands = nodes(arena, operands.count, noperands),
                .type_arguments = instruction->payload.prim_op.type_arguments,
            }));
            instruction = yield_values_and_wrap_in_block(bb, results);
            return let(arena, instruction, tail);
        }
        case Block_TAG: {
            // follow the terminator of the block until we hit a yield()
            const Node* lam = instruction->payload.block.inside;
            const Node* terminator = get_abstraction_body(lam);
            size_t depth = 0;
            bool dry_run = true;
            const Node** lets = NULL;
            while (true) {
                assert(is_case(lam));
                switch (is_terminator(terminator)) {
                    case NotATerminator: assert(false);
                    case Terminator_Let_TAG: {
                        if (lets)
                            lets[depth] = terminator;
                        lam = get_let_tail(terminator);
                        terminator = get_abstraction_body(lam);
                        depth++;
                        continue;
                    }
                    case Terminator_Yield_TAG: {
                        if (dry_run) {
                            lets = calloc(sizeof(const Node*), depth);
                            dry_run = false;
                            depth = 0;
                            // Start over !
                            lam = instruction->payload.block.inside;
                            terminator = get_abstraction_body(lam);
                            continue;
                        } else {
                            // wrap the original tail with the args of join()
                            assert(is_case(tail));
                            const Node* acc = let(arena, quote_helper(arena, terminator->payload.yield.args), tail);
                            // rebuild the let chain that we traversed
                            for (size_t i = 0; i < depth; i++) {
                                const Node* olet = lets[depth - 1 - i];
                                const Node* olam = get_let_tail(olet);
                                assert(olam->tag == Case_TAG);
                                Nodes params = get_abstraction_params(olam);
                                for (size_t j = 0; j < params.count; j++) {
                                    // recycle the params by setting their abs value to NULL
                                    *((Node**) &(params.nodes[j]->payload.var.abs)) = NULL;
                                }
                                const Node* nlam = case_(arena, params, acc);
                                acc = let(arena, get_let_instruction(olet), nlam);
                            }
                            free(lets);
                            return acc;
                        }
                    }
                    // if we see anything else, give up
                    default: {
                        assert(dry_run);
                        return node;
                    }
                }
            }
        }
        default: break;
    }

    return node;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    PrimOp payload = node->payload.prim_op;

    LARRAY(const IntLiteral*, int_literals, payload.operands.count);
    bool all_int_literals = true;
    IntSizes width;
    bool is_signed;
    for (size_t i = 0; i < payload.operands.count; i++) {
        int_literals[i] = resolve_to_literal(payload.operands.nodes[i]);
        all_int_literals &= int_literals[i] != NULL;
        if (int_literals[i]) {
            const Type* int_t = payload.operands.nodes[i]->type;
            deconstruct_qualified_type(&int_t);
            assert(int_t->tag == Int_TAG);
            width = int_t->payload.int_type.width;
            is_signed = int_t->payload.int_type.is_signed;
        }
    }

    switch (payload.op) {
        case add_op: {
            if (all_int_literals)
                return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value + int_literals[1]->value }));
            // If either operand is zero, destroy the add
            for (size_t i = 0; i < 2; i++)
                if (is_zero(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[1 - i]);
            break;
        }
        case sub_op: {
            if (all_int_literals)
                return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value - int_literals[1]->value }));
            // If second operand is zero, return the first one
            if (is_zero(payload.operands.nodes[1]))
                return quote_single(arena, payload.operands.nodes[0]);
            // if first operand is zero, invert the second one
            if (is_zero(payload.operands.nodes[0]))
                return prim_op(arena, (PrimOp) { .op = neg_op, .operands = singleton(payload.operands.nodes[1]), .type_arguments = empty(arena) });
            break;
        }
        case mul_op: {
            if (all_int_literals) {
                if (is_signed)
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value * int_literals[1]->value }));
                else
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value * int_literals[1]->value }));
            }
            for (size_t i = 0; i < 2; i++)
                if (is_zero(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[i]); // return zero !

            for (size_t i = 0; i < 2; i++)
                if (is_one(payload.operands.nodes[i]))
                    return quote_single(arena, payload.operands.nodes[1 - i]);

            break;
        }
        case div_op: {
            if (all_int_literals) {
                if (is_signed)
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value / int_literals[1]->value }));
                else
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = width, .value = int_literals[0]->value / int_literals[1]->value }));
            }
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
        case reinterpret_op:
        case convert_op:
            // get rid of identity casts
            if (payload.type_arguments.nodes[0] == get_unqualified_type(payload.operands.nodes[0]->type))
                return quote_single(arena, payload.operands.nodes[0]);
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

const Node* fold_node(IrArena* arena, const Node* node) {
    const Node* folded = node;
    switch (node->tag) {
        case Let_TAG: folded = fold_let(arena, node); break;
        case PrimOp_TAG: folded = fold_prim_op(arena, node); break;
        case Block_TAG: {
            const Node* lam = node->payload.block.inside;
            const Node* body = lam->payload.case_.body;
            if (body->tag == Yield_TAG) {
                return quote_helper(arena, body->payload.yield.args);
            } else if (body->tag == Let_TAG) {
                // fold block { let x, y, z = I; yield (x, y, z); } back to I
                const Node* instr = get_let_instruction(body);
                const Node* let_case = get_let_tail(body);
                const Node* let_case_body = get_abstraction_body(let_case);
                if (let_case_body->tag == Yield_TAG) {
                    bool only_forwards = true;
                    Nodes let_case_params = get_abstraction_params(let_case);
                    Nodes yield_args = let_case_body->payload.yield.args;
                    if (let_case_params.count == yield_args.count) {
                        for (size_t i = 0; i < yield_args.count; i++) {
                            only_forwards &= yield_args.nodes[i] == let_case_params.nodes[i];
                        }
                        if (only_forwards) {
                            debugv_print("Fold: simplify ");
                            log_node(DEBUGV, node);
                            debugv_print(" into just ");
                            log_node(DEBUGV, instr);
                            debugv_print(".\n");
                            return instr;
                        }
                    }
                }
            }
            break;
        }
        case AntiQuote_TAG: {
            const Node* instr = node->payload.anti_quote.instruction;
            if (instr->tag == PrimOp_TAG && instr->payload.prim_op.op == quote_op) {
                assert(instr->payload.prim_op.operands.count == 1);
                return first(instr->payload.prim_op.operands);
            }
            break;
        }
        case If_TAG: {
            If payload = node->payload.if_instr;
            const Node* false_case = payload.if_false;
            if (arena->config.optimisations.delete_unreachable_structured_cases && false_case && is_unreachable_case(false_case))
                return block(arena, (Block) { .inside = payload.if_true, .yield_types = add_qualifiers(arena, payload.yield_types, false) });
            break;
        }
        case Match_TAG: {
            if (!arena->config.optimisations.delete_unreachable_structured_cases)
                break;
            Match payload = node->payload.match_instr;
            Nodes old_cases = payload.cases;
            LARRAY(const Node*, literals, old_cases.count);
            LARRAY(const Node*, cases, old_cases.count);
            size_t new_cases_count = 0;
            for (size_t i = 0; i < old_cases.count; i++) {
                const Node* c = old_cases.nodes[i];
                if (is_unreachable_case(c))
                    continue;
                literals[new_cases_count] = node->payload.match_instr.literals.nodes[i];
                cases[new_cases_count] = node->payload.match_instr.cases.nodes[i];
                new_cases_count++;
            }
            if (new_cases_count == old_cases.count)
                break;

            if (new_cases_count == 1 && is_unreachable_case(payload.default_case))
                return block(arena, (Block) { .inside = cases[0], .yield_types = add_qualifiers(arena, payload.yield_types, false) });

            if (new_cases_count == 0)
                return block(arena, (Block) { .inside = payload.default_case, .yield_types = add_qualifiers(arena, payload.yield_types, false) });

            return match_instr(arena, (Match) {
                .inspect = payload.inspect,
                .yield_types = payload.yield_types,
                .default_case = payload.default_case,
                .literals = nodes(arena, new_cases_count, literals),
                .cases = nodes(arena, new_cases_count, cases),
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
