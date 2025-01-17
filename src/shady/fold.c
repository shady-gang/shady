#include "fold.h"

#include "shady/ir/memory_layout.h"

#include "check.h"

#include "portability.h"

#include <assert.h>
#include <math.h>

static const Node* quote_single(IrArena* a, const Node* value) {
    return value;
}

static bool is_zero(const Node* node) {
    const IntLiteral* lit = shd_resolve_to_int_literal(node);
    if (lit && shd_get_int_literal_value(*lit, false) == 0)
        return true;
    return false;
}

static bool is_one(const Node* node) {
    const IntLiteral* lit = shd_resolve_to_int_literal(node);
    if (lit && shd_get_int_literal_value(*lit, false) == 1)
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
        int_literals[i] = shd_resolve_to_int_literal(payload.operands.nodes[i]);
        all_int_literals &= int_literals[i] != NULL;
        if (int_literals[i]) {
            int_width = int_literals[i]->width;
            is_signed = int_literals[i]->is_signed;
        }

        float_literals[i] = shd_resolve_to_float_literal(payload.operands.nodes[i]);
        if (float_literals[i])
            float_width = float_literals[i]->width;
        all_float_literals &= float_literals[i] != NULL;
    }

#define UN_OP(primop, op) case primop##_op: \
if (all_int_literals)        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = int_width, .value = op int_literals[0]->value})); \
else if (all_float_literals) return quote_single(arena, shd_fp_literal_helper(arena, float_width, op shd_get_float_literal_value(*float_literals[0]))); \
else break;

#define BIN_OP(primop, op) case primop##_op: \
if (all_int_literals)        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = is_signed, .width = int_width, .value = int_literals[0]->value op int_literals[1]->value })); \
else if (all_float_literals) return quote_single(arena, shd_fp_literal_helper(arena, float_width, shd_get_float_literal_value(*float_literals[0]) op shd_get_float_literal_value(*float_literals[1]))); \
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
                    return quote_single(arena, shd_fp_literal_helper(arena, float_width, fmod(shd_get_float_literal_value(*float_literals[0]), shd_get_float_literal_value(*float_literals[1]))));
            case reinterpret_op: {
                const Type* dst_t = shd_first(payload.type_arguments);
                uint64_t raw_value = int_literals[0] ? int_literals[0]->value : float_literals[0]->value;
                if (dst_t->tag == Int_TAG) {
                    return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = raw_value }));
                } else if (dst_t->tag == Float_TAG) {
                    return quote_single(arena, float_literal(arena, (FloatLiteral) { .width = dst_t->payload.float_type.width, .value = raw_value }));
                }
                break;
            }
            case convert_op: {
                const Type* dst_t = shd_first(payload.type_arguments);
                uint64_t bitmask = 0;
                if (shd_get_type_bitwidth(dst_t) == 64)
                    bitmask = UINT64_MAX;
                else
                    bitmask = ~(UINT64_MAX << shd_get_type_bitwidth(dst_t));
                if (dst_t->tag == Int_TAG) {
                    if (all_int_literals) {
                        uint64_t old_value = shd_get_int_literal_value(*int_literals[0], int_literals[0]->is_signed);
                        uint64_t value = old_value & bitmask;
                        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value }));
                    } else if (all_float_literals) {
                        double old_value = shd_get_float_literal_value(*float_literals[0]);
                        int64_t value = old_value;
                        return quote_single(arena, int_literal(arena, (IntLiteral) { .is_signed = dst_t->payload.int_type.is_signed, .width = dst_t->payload.int_type.width, .value = value }));
                    }
                } else if (dst_t->tag == Float_TAG) {
                    if (all_int_literals) {
                        uint64_t old_value = shd_get_int_literal_value(*int_literals[0], int_literals[0]->is_signed);
                        double value = old_value;
                        return quote_single(arena, shd_fp_literal_helper(arena, dst_t->payload.float_type.width, value));
                    } else if (all_float_literals) {
                        double old_value = shd_get_float_literal_value(*float_literals[0]);
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
                return prim_op(arena, (PrimOp) { .op = neg_op, .operands = shd_singleton(payload.operands.nodes[1]), .type_arguments = shd_empty(arena) });
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

/**
 * The high-level goal is to specialize away reinterpret/convert ops by generally pushing them "outwards" and deleting them where possible
 * For "Lea" ops, we can move the conversion to generic after the OP itself but the pointee type must match for address calculations to remain identitical
 * For "Access" (Load,Store,Copy,Fill) ops, we just need to ensure compatible pointers for well-defined results
 * For "Convert" op we can move any reinterpret ops we encounter on the outer level
 */
static inline const Node* resolve_ptr_source0(const Node* ptr, bool ensure_compatible_pointee) {
    const Node* original_ptr = ptr;

    while (true) {
        const Type* ptr_type = shd_get_unqualified_type(ptr->type);
        assert(ptr_type->tag == PtrType_TAG);
        PtrType ptr_type_payload = ptr_type->payload.ptr_type;
        switch (ptr->tag) {
            case PrimOp_TAG: {
                PrimOp instruction = ptr->payload.prim_op;
                // casts starting from non-ptrs are not elligible
                const Node* src = shd_first(instruction.operands);
                if (shd_get_unqualified_type(src->type)->tag != PtrType_TAG)
                    break;
                switch (instruction.op) {
                    case reinterpret_op: {
                        if (ensure_compatible_pointee)
                            break;
                        ptr = shd_first(instruction.operands);
                        continue;
                    }
                    case convert_op: {
                        // only ptr-to-generic pointers are acceptable
                        if (shd_first(instruction.type_arguments)->tag != PtrType_TAG)
                            break;
                        ptr = src;
                        continue;
                    }
                    default: break;
                }
                break;
            }
            case PtrCompositeElement_TAG: {
                PtrCompositeElement payload = ptr->payload.ptr_composite_element;
                if (is_zero(payload.index) && !ptr_type_payload.is_reference && !ensure_compatible_pointee) {
                    ptr = payload.ptr;
                    continue;
                }
                break;
            }
            default: break;
        }
        break;
    }

    if (ptr != original_ptr)
        return ptr;
    return NULL;
}

static bool is_generic(const Node* ptr) {
    const Type* t = shd_get_unqualified_type(ptr->type);
    assert(t->tag == PtrType_TAG);
    return t->payload.ptr_type.address_space == AsGeneric;
}

static const Type* make_ptr_generic(const Type* old) {
    PtrType payload = old->payload.ptr_type;
    payload.address_space = AsGeneric;
    return ptr_type(old->arena, payload);
}

static const Type* change_pointee(const Type* old, const Type* pointee) {
    PtrType payload = old->payload.ptr_type;
    payload.pointed_type = pointee;
    return ptr_type(old->arena, payload);
}

static inline const Node* resolve_ptr_source(const Node* ptr, bool ensure_compatible_pointee) {
    IrArena* arena = ptr->arena;
    if (!ensure_compatible_pointee)
        return resolve_ptr_source0(ptr, false);
    const Node* ptr_non_compatible = resolve_ptr_source0(ptr, false);
    const Node* ptr_compatible = resolve_ptr_source0(ptr, true);
    // if going for an incompatible pointer doesn't simplify anything
    if (!ptr_non_compatible || ptr_non_compatible == ptr_compatible)
        return ptr_compatible;

    if (is_generic(ptr) && !is_generic(ptr_non_compatible)) {
        const Node* r = ptr_non_compatible;
        // r = prim_op_helper(arena, convert_op, shd_singleton(make_ptr_generic(shd_get_unqualified_type(r->type))),shd_singleton(r));
        const Node* dst_t = change_pointee(shd_get_unqualified_type(r->type), shd_get_unqualified_type(ptr->type)->payload.ptr_type.pointed_type);
        r = prim_op_helper(arena, reinterpret_op, shd_singleton(dst_t),shd_singleton(r));
        return r;
    }

    return NULL;
}

static void maybe_convert_to_generic(const Node* old, const Node** new) {
    IrArena* arena = old->arena;
    const Type* old_t = shd_get_unqualified_type(old->type);
    assert(old_t->tag == PtrType_TAG);
    if (old_t->payload.ptr_type.address_space == AsGeneric)
        *new = prim_op_helper(arena, convert_op, shd_singleton(make_ptr_generic(shd_get_unqualified_type((*new)->type))),shd_singleton(*new));
}

static const Node* to_ptr_size(const Node* n) {
    IrArena* a = n->arena;
    return prim_op_helper(a, convert_op, shd_singleton(shd_uint64_type(a)), shd_singleton(n));
}

static inline const Node* fold_simplify_ptr_operand(const Node* node) {
    IrArena* arena = node->arena;
    const Node* r = NULL;
    switch (node->tag) {
        case PrimOp_TAG: {
            PrimOp payload = node->payload.prim_op;
            switch (payload.op) {
                case convert_op: {
                    const Type* dst_t = payload.type_arguments.nodes[0];
                    if (dst_t->tag != PtrType_TAG || dst_t->payload.ptr_type.address_space != AsGeneric)
                        break;
                    // only bother with Generic casts
                    const Node* src = resolve_ptr_source(shd_first(payload.operands), true);
                    const Node* nptr = src;
                    if (nptr) {
                        r = prim_op_helper(arena, convert_op, shd_singleton(make_ptr_generic(shd_get_unqualified_type(nptr->type))),shd_singleton(nptr));
                        //r = prim_op_helper(arena, reinterpret_op, shd_singleton(shd_get_unqualified_type(node->type)),shd_singleton(r));
                    }
                    break;
                }
                default: break;
            }
            break;
        }
        case Load_TAG: {
            Load payload = node->payload.load;
            const Node* nptr = resolve_ptr_source(payload.ptr, true);
            if (!nptr) break;
            payload.ptr = nptr;
            r = load(arena, payload);
            break;
        }
        case Store_TAG: {
            Store payload = node->payload.store;
            const Node* nptr = resolve_ptr_source(payload.ptr, true);
            if (!nptr) break;
            payload.ptr = nptr;
            r = store(arena, payload);
            break;
        }
        case PtrCompositeElement_TAG: {
            PtrCompositeElement payload = node->payload.ptr_composite_element;
            const Node* nptr = resolve_ptr_source(payload.ptr, true);
            if (!nptr) break;
            payload.ptr = nptr;
            r = ptr_composite_element(arena, payload);
            maybe_convert_to_generic(node, &r);
            break;
        }
        case PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset payload = node->payload.ptr_array_element_offset;
            if (is_zero(payload.offset))
                return payload.ptr;
            const Node* nptr = resolve_ptr_source(payload.ptr, true);
            if (!nptr) break;
            payload.ptr = nptr;
            if (nptr->tag == PtrCompositeElement_TAG) {
                PtrCompositeElement other_offset = nptr->payload.ptr_composite_element;
                payload.ptr = other_offset.ptr;
                other_offset.index = prim_op_helper(arena, add_op, shd_empty(arena), mk_nodes(arena, to_ptr_size(other_offset.index), to_ptr_size(payload.offset)));
                r = ptr_composite_element(arena, other_offset);
                maybe_convert_to_generic(node, &r);
                break;
            }
            r = ptr_array_element_offset(arena, payload);
            maybe_convert_to_generic(node, &r);
            break;
        }
        case IndirectCall_TAG: {
            IndirectCall payload = node->payload.indirect_call;
            if (payload.callee->tag == FnAddr_TAG)
                return call_helper(arena, payload.mem, payload.callee->payload.fn_addr.fn, payload.args);
            const Node* nptr = resolve_ptr_source(payload.callee, true);
            if (!nptr) break;
            payload.callee = nptr;
            r = indirect_call(arena, payload);
            break;
        }
        case TailCall_TAG: {
            TailCall payload = node->payload.tail_call;
            const Node* nptr = resolve_ptr_source(payload.callee, true);
            if (!nptr) break;
            payload.callee = nptr;
            r = tail_call(arena, payload);
            break;
        }
        default: return node;
    }

    if (!r)
        return node;

    // if (!shd_is_subtype(node->type, r->type))
    //     r = prim_op_helper(arena, convert_op, shd_singleton(shd_get_unqualified_type(node->type)), shd_singleton(r));
    return r;
}

static const Node* fold_prim_op(IrArena* arena, const Node* node) {
    APPLY_FOLD(fold_constant_math)
    APPLY_FOLD(fold_simplify_math)

    PrimOp payload = node->payload.prim_op;
    switch (payload.op) {
        // TODO: case subgroup_broadcast_first_op:
        case subgroup_assume_uniform_op: {
            const Node* value = shd_first(payload.operands);
            if (shd_is_qualified_type_uniform(value->type))
                return quote_single(arena, value);
            break;
        }
        case convert_op:
        case reinterpret_op: {
            // get rid of identity casts
            if (payload.type_arguments.nodes[0] == shd_get_unqualified_type(payload.operands.nodes[0]->type))
                return quote_single(arena, payload.operands.nodes[0]);
            // reinterpret[A](reinterpret[B](x)) => reinterpret[A](x)
            if (payload.op == reinterpret_op) {
                const Node* src = shd_first(payload.operands);
                if (src->tag == PrimOp_TAG && src->payload.prim_op.op == reinterpret_op) {
                    payload.operands = shd_singleton(shd_first(src->payload.prim_op.operands));
                    return prim_op(arena, payload);
                }
                // over-fit hack for LLVM output:
                if (shd_first(payload.type_arguments)->tag == PtrType_TAG && shd_get_unqualified_type(src->type)->tag == PtrType_TAG && arena->config.optimisations.weaken_bitcast_to_lea) {
                    const Type* src_type = shd_get_pointer_type_element(shd_get_unqualified_type(src->type));
                    if (src_type->tag == NominalType_TAG)
                        src_type = src_type->payload.nom_type.body;
                    const Type* dst_type = shd_get_pointer_type_element(shd_get_unqualified_type(node->type));
                    if (src_type->tag == RecordType_TAG && src_type->payload.record_type.members.count > 0) {
                        if (src_type->payload.record_type.members.nodes[0] == dst_type) {
                            return ptr_composite_element_helper(arena, src, shd_uint32_literal(arena, 0));
                        }
                    } else if (src_type->tag == PackType_TAG) {
                        if (src_type->payload.pack_type.element_type == dst_type) {
                            return ptr_composite_element_helper(arena, src, shd_uint32_literal(arena, 0));
                        }
                    } else if (src_type->tag == ArrType_TAG) {
                        if (src_type->payload.arr_type.element_type == dst_type) {
                            return ptr_composite_element_helper(arena, src, shd_uint32_literal(arena, 0));
                        }
                    }
                }
            }
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
                return mem_and_value(arena, (MemAndValue) { .value = undef(arena, (Undef) { .type = shd_get_unqualified_type(node->type) }), .mem = node->payload.load.mem });
            break;
        }
        case Store_TAG: {
            if (node->payload.store.ptr->tag == Undef_TAG)
                return node->payload.store.mem;
                // return mem_and_value(arena, (MemAndValue) { .value = undef_helper(arena, node->type), .mem = node->payload.store.mem });
                // return mem_and_value(arena, (MemAndValue) { .value = empty_multiple_return_value(arena), .mem = node->payload.store.mem });
                // return mem_and_value(arena, (MemAndValue) { .value = shd_tuple_helper(arena, shd_empty(arena)), .mem = node->payload.store.mem });
            break;
        }
        case PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset payload = node->payload.ptr_array_element_offset;
            if (payload.ptr->tag == Undef_TAG)
                return quote_single(arena, undef(arena, (Undef) { .type = shd_get_unqualified_type(node->type) }));
            break;
        }
        case PtrCompositeElement_TAG: {
            PtrCompositeElement payload = node->payload.ptr_composite_element;
            if (payload.ptr->tag == Undef_TAG)
                return quote_single(arena, undef(arena, (Undef) { .type = shd_get_unqualified_type(node->type) }));
            break;
        }
        case PrimOp_TAG: {
            PrimOp payload = node->payload.prim_op;
            switch (payload.op) {
                case reinterpret_op:
                case convert_op: {
                    if (shd_first(payload.operands)->tag == Undef_TAG)
                        return quote_single(arena, undef(arena, (Undef) { .type = shd_get_unqualified_type(node->type) }));
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

const Node* _shd_fold_node(IrArena* arena, const Node* node) {
    const Node* const original_node = node;
    node = fold_memory_poison(arena, node);
    node = fold_simplify_ptr_operand(node);
    switch (node->tag) {
        case PrimOp_TAG: node = fold_prim_op(arena, node); break;
        case PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset payload = node->payload.ptr_array_element_offset;
            if (is_zero(payload.offset))
                return payload.ptr;
            break;
        }
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
                .literals = shd_nodes(arena, new_cases_count, literals),
                .cases = shd_nodes(arena, new_cases_count, cases),
                .tail = payload.tail,
                .mem = payload.mem,
            });
        }
        default: break;
    }

    // catch bad folding rules that mess things up
    if (is_value(original_node)) assert(is_value(node));
    // if (is_instruction(original_node)) assert(is_instruction(node) || is_value(node));
    if (is_terminator(original_node)) assert(is_terminator(node));

    if (original_node->type && shd_is_value_type(original_node->type))
        assert(shd_is_subtype(original_node->type, node->type));

    return node;
}

const Node* _shd_fold_node_operand(NodeTag tag, NodeClass nc, String opname, const Node* op) {
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
