#include "fold.h"

#include "log.h"

#include "type.h"
#include "portability.h"
#include "rewrite.h"

#include "transform/ir_gen_helpers.h"

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

#define APPLY_FOLD(F) { const Node* applied_fold = F(node); if (applied_fold) return applied_fold; }

static inline const Node* fold_constant_math(const Node* node) {
    IrArena* arena = node->arena;
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

    return NULL;
}

static inline const Node* fold_simplify_math(const Node* node) {
    IrArena* arena = node->arena;
    PrimOp payload = node->payload.prim_op;
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
                return prim_op(arena, (PrimOp) {.op = neg_op, .operands = singleton(payload.operands.nodes[1]), .type_arguments = empty(arena)});
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
        case eq_op: {
            if (payload.operands.nodes[0] == payload.operands.nodes[1])
                return quote_single(arena, true_lit(arena));
            break;
        }
        case neq_op: {
            if (payload.operands.nodes[0] == payload.operands.nodes[1])
                return quote_single(arena, false_lit(arena));
            break;
        }
        default: break;
    }

    return NULL;
}

static inline const Node* resolve_ptr_source(BodyBuilder* bb, const Node* ptr) {
    const Node* original_ptr = ptr;
    IrArena* a = ptr->arena;
    const Type* t = ptr->type;
    bool u = deconstruct_qualified_type(&t);
    assert(t->tag == PtrType_TAG);
    const Type* desired_pointee_type = t->payload.ptr_type.pointed_type;
    // const Node* last_known_good = node;

    int distance = 0;
    bool specialize_generic = false;
    AddressSpace src_as = t->payload.ptr_type.address_space;
    while (ptr->tag == Variablez_TAG) {
        const Node* def = get_var_def(ptr->payload.varz);
        if (def->tag != PrimOp_TAG)
            break;
        PrimOp instruction = def->payload.prim_op;
        switch (instruction.op) {
            case reinterpret_op: {
                distance++;
                ptr = first(instruction.operands);
                continue;
            }
            case convert_op: {
                // only conversions to generic pointers are acceptable
                if (first(instruction.type_arguments)->tag != PtrType_TAG)
                    break;
                assert(!specialize_generic && "something should not be converted to generic twice!");
                specialize_generic = true;
                ptr = first(instruction.operands);
                src_as = get_unqualified_type(ptr->type)->payload.ptr_type.address_space;
                continue;
            }
            case lea_op: {
                Nodes ops = instruction.operands;
                for (size_t i = 1; i < ops.count; i++) {
                    if (!is_zero(ops.nodes[i]))
                        goto outer_break;
                }
                distance++;
                ptr = first(ops);
                continue;
                outer_break:
                break;
            }
            default: break;
        }
        break;
    }

    // if there was more than one of those pointless casts...
    if (distance > 1 || specialize_generic) {
        const Type* new_src_ptr_type = ptr->type;
        deconstruct_qualified_type(&new_src_ptr_type);
        if (new_src_ptr_type->tag != PtrType_TAG || new_src_ptr_type->payload.ptr_type.pointed_type != desired_pointee_type) {
            PtrType payload = t->payload.ptr_type;
            payload.address_space = src_as;
            ptr = gen_reinterpret_cast(bb, ptr_type(a, payload), ptr);
        }
        return ptr;
    }
    return original_ptr;
}

static void inline simplify_ptr_operand(IrArena* a, BodyBuilder* bb, PrimOp* payload, bool* success, int i) {
    const Node* old_op = payload->operands.nodes[i];
    const Type* ptr_t = old_op->type;
    deconstruct_qualified_type(&ptr_t);
    if (ptr_t->payload.ptr_type.is_reference)
        return;
    const Node* new_op = resolve_ptr_source(bb, old_op);
    if (old_op != new_op) {
        payload->operands = change_node_at_index(a, payload->operands, i, new_op);
        *success = true;
    }
}

static inline const Node* fold_simplify_ptr_operand(const Node* node) {
    IrArena* arena = node->arena;
    PrimOp payload = node->payload.prim_op;
    BodyBuilder* bb = begin_body(arena);
    bool rebuild = false;
    switch (payload.op) {
        case store_op:
        case load_op:
        case lea_op: {
            simplify_ptr_operand(arena, bb, &payload, &rebuild, 0);
            break;
        }
        default: break;
    }

    if (rebuild) {
        const Node* r = prim_op(arena, payload);
        if (!is_subtype(node->type, r->type)) {
            r = gen_conversion(bb, get_unqualified_type(node->type), first(bind_instruction(bb, r)));
            return yield_values_and_wrap_in_block(bb, singleton(r));
        }
        return bind_last_instruction_and_wrap_in_block(bb, r);
    }

    cancel_body(bb);
    return NULL;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    APPLY_FOLD(fold_constant_math)
    APPLY_FOLD(fold_simplify_math)
    APPLY_FOLD(fold_simplify_ptr_operand)

    PrimOp payload = node->payload.prim_op;
    switch (payload.op) {
        case subgroup_assume_uniform_op:
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

const Node* fold_node(IrArena* arena, const Node* node) {
    const Node* folded = node;
    switch (node->tag) {
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
        case If_TAG: {
            If payload = node->payload.if_instr;
            const Node* false_case = payload.if_false;
            if (arena->config.optimisations.delete_unreachable_structured_cases && false_case && is_unreachable_case(false_case))
                return block(arena, (Block) { .inside = payload.if_true, .yield_types = add_qualifiers(arena, payload.yield_types, false) });
            if (arena->config.optimisations.fold_static_control_flow) {
                if (payload.condition == true_lit(arena)) {
                    return block(arena, (Block) { .inside = payload.if_true, .yield_types = add_qualifiers(arena, payload.yield_types, false) });
                } else if (payload.condition == false_lit(arena) && false_case) {
                    return block(arena, (Block) { .inside = false_case, .yield_types = add_qualifiers(arena, payload.yield_types, false) });
                }
            }
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
