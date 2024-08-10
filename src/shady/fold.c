#include "fold.h"

#include "log.h"

#include "type.h"
#include "portability.h"
#include "rewrite.h"

#include "transform/ir_gen_helpers.h"

#include <assert.h>
#include <math.h>

static const Node* quote_single(IrArena* a, const Node* value) {
    return value;
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
                        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value }));
                    } else if (all_float_literals) {
                        double old_value = get_float_literal_value(*float_literals[0]);
                        int64_t value = old_value;
                        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value }));
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

static inline const Node* resolve_ptr_source(const Node* ptr) {
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
    while (true) {
        const Node* def = ptr;
        switch (def->tag) {
            case PrimOp_TAG: {
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
                    default: break;
                }
                break;
            }
            case Lea_TAG: {
                Lea payload = def->payload.lea;
                if (!is_zero(payload.offset))
                    goto outer_break;
                for (size_t i = 0; i < payload.indices.count; i++) {
                    if (!is_zero(payload.indices.nodes[i]))
                        goto outer_break;
                }
                distance++;
                ptr = payload.ptr;
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
            ptr = prim_op_helper(a, reinterpret_op, singleton(ptr_type(a, payload)), singleton(ptr));
        }
        return ptr;
    }
    return NULL;
}

static inline const Node* simplify_ptr_operand(IrArena* a, const Node* old_op) {
    const Type* ptr_t = old_op->type;
    deconstruct_qualified_type(&ptr_t);
    if (ptr_t->payload.ptr_type.is_reference)
        return NULL;
    return resolve_ptr_source(old_op);
}

static inline const Node* fold_simplify_ptr_operand(const Node* node) {
    IrArena* arena = node->arena;
    const Node* r = NULL;
    switch (node->tag) {
        case Load_TAG: {
            Load payload = node->payload.load;
            const Node* nptr = simplify_ptr_operand(arena, payload.ptr);
            if (!nptr) break;
            payload.ptr = nptr;
            r = load(arena, payload);
            break;
        }
        case Store_TAG: {
            Store payload = node->payload.store;
            const Node* nptr = simplify_ptr_operand(arena, payload.ptr);
            if (!nptr) break;
            payload.ptr = nptr;
            r = store(arena, payload);
            break;
        }
        case Lea_TAG: {
            Lea payload = node->payload.lea;
            const Node* nptr = simplify_ptr_operand(arena, payload.ptr);
            if (!nptr) break;
            payload.ptr = nptr;
            r = lea(arena, payload);
            break;
        }
        default: return node;
    }

    if (!r)
        return node;

    if (!is_subtype(node->type, r->type))
        r = prim_op_helper(arena, convert_op, singleton(get_unqualified_type(node->type)), singleton(r));
    return r;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    APPLY_FOLD(fold_constant_math)
    APPLY_FOLD(fold_simplify_math)

    PrimOp payload = node->payload.prim_op;
    switch (payload.op) {
        // TODO: case subgroup_broadcast_first_op:
        case subgroup_assume_uniform_op: {
            const Node* value = first(payload.operands);
            if (is_qualified_type_uniform(value->type))
                return quote_single(arena, value);
            break;
        }
        case convert_op:
        case reinterpret_op: {
            // get rid of identity casts
            if (payload.type_arguments.nodes[0] == get_unqualified_type(payload.operands.nodes[0]->type))
                return quote_single(arena, payload.operands.nodes[0]);
            break;
        }
        default: break;
    }
    return node;
}

static const Node* fold_memory_poison(IrArena* arena, const Node* node) {
    switch (node->tag) {
        case Load_TAG: {
            if (node->payload.load.ptr->tag == Undef_TAG)
                return mem_and_value(arena, (MemAndValue) { .value = undef(arena, (Undef) { .type = get_unqualified_type(node->type) }), .mem = node->payload.load.mem });
            break;
        }
        case Store_TAG: {
            if (node->payload.store.ptr->tag == Undef_TAG)
                return mem_and_value(arena, (MemAndValue) { .value = tuple_helper(arena, empty(arena)), .mem = node->payload.store.mem });
            break;
        }
        case Lea_TAG: {
            Lea payload = node->payload.lea;
            if (payload.ptr->tag == Undef_TAG)
                return quote_single(arena, undef(arena, (Undef) { .type = get_unqualified_type(node->type) }));
            break;
        }
        case PrimOp_TAG: {
            PrimOp payload = node->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op:
                case convert_op: {
                    if (first(payload.operands)->tag == Undef_TAG)
                        return quote_single(arena, undef(arena, (Undef) { .type = get_unqualified_type(node->type) }));
                    break;
                }
                default: break;
            }
            break;
        }
        default: break;
    }
    return node;
}

static bool is_unreachable_case(const Node* c) {
    assert(c && c->tag == BasicBlock_TAG);
    const Node* b = get_abstraction_body(c);
    return b->tag == Unreachable_TAG;
}

static bool is_unreachable_destination(const Node* j) {
    assert(j && j->tag == Jump_TAG);
    const Node* b = get_abstraction_body(j->payload.jump.target);
    return b->tag == Unreachable_TAG;
}

const Node* fold_node(IrArena* arena, const Node* node) {
    const Node* const original_node = node;
    node = fold_memory_poison(arena, node);
    node = fold_simplify_ptr_operand(node);
    switch (node->tag) {
        case PrimOp_TAG: node = fold_prim_op(arena, node); break;
        case Branch_TAG: {
            Branch payload = node->payload.branch;
            if (arena->config.optimisations.fold_static_control_flow) {
                if (payload.condition == true_lit(arena)) {
                    return payload.true_jump;
                } else if (payload.condition == false_lit(arena)) {
                    return payload.false_jump;
                }
            } else if (arena->config.optimisations.delete_unreachable_structured_cases) {
                if (is_unreachable_destination(payload.true_jump))
                    return payload.false_jump;
                else if (is_unreachable_destination(payload.false_jump))
                    return payload.true_jump;
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

            /*if (new_cases_count == 1 && is_unreachable_case(payload.default_case))
                return block(arena, (Block) { .inside = cases[0], .yield_types = add_qualifiers(arena, payload.yield_types, false) });

            if (new_cases_count == 0)
                return block(arena, (Block) { .inside = payload.default_case, .yield_types = add_qualifiers(arena, payload.yield_types, false) });*/

            return match_instr(arena, (Match) {
                .inspect = payload.inspect,
                .yield_types = payload.yield_types,
                .default_case = payload.default_case,
                .literals = nodes(arena, new_cases_count, literals),
                .cases = nodes(arena, new_cases_count, cases),
                .tail = payload.tail,
            });
        }
        default: break;
    }

    // catch bad folding rules that mess things up
    if (is_value(original_node)) assert(is_value(node));
    if (is_instruction(original_node)) assert(is_instruction(node) || is_value(node));
    if (is_terminator(original_node)) assert(is_terminator(node));

    if (node->type)
        assert(is_subtype(original_node->type, node->type));

    return node;
}

const Node* fold_node_operand(NodeTag tag, NodeClass nc, String opname, const Node* op) {
    if (!op)
        return NULL;
    if (op->tag == MemAndValue_TAG) {
        MemAndValue payload = op->payload.mem_and_value;
        if (nc == NcMem)
            return payload.mem;
        return payload.value;
    }
    return op;
}
