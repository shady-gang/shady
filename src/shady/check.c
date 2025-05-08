#include "check.h"
#include "shady/ir/cast.h"

#include "log.h"
#include "ir_private.h"
#include "portability.h"
#include "dict.h"
#include "util.h"

#include "shady/ir/builtin.h"

#include <string.h>
#include <assert.h>

static bool are_types_identical(size_t num_types, const Type* types[]) {
    for (size_t i = 0; i < num_types; i++) {
        assert(types[i]);
        if (types[0] != types[i])
            return false;
    }
    return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

const Type* _shd_check_type_join_point_type(IrArena* arena, JoinPointType type) {
    for (size_t i = 0; i < type.yield_types.count; i++) {
        assert(shd_is_data_type(type.yield_types.nodes[i]));
    }
    return NULL;
}

const Type* _shd_check_type_record_type(IrArena* arena, RecordType type) {
    assert(type.names.count == 0 || type.names.count == type.members.count);
    for (size_t i = 0; i < type.members.count; i++) {
        // member types are value types iff this is a return tuple
        if (type.special == MultipleReturn)
            assert(shd_is_value_type(type.members.nodes[i]));
        else
            assert(shd_is_data_type(type.members.nodes[i]));
    }
    return NULL;
}

const Type* _shd_check_type_qualified_type(IrArena* arena, QualifiedType qualified_type) {
    assert(shd_is_data_type(qualified_type.type));
    return NULL;
}

const Type* _shd_check_type_arr_type(IrArena* arena, ArrType type) {
    assert(shd_is_data_type(type.element_type));
    return NULL;
}

const Type* _shd_check_type_vector_type(IrArena* arena, VectorType vector_type) {
    assert(shd_is_data_type(vector_type.element_type));
    return NULL;
}

const Type* _shd_check_type_ptr_type(IrArena* arena, PtrType ptr_type) {
    if (!arena->config.target.memory.address_spaces[ptr_type.address_space].allowed) {
        shd_error_print("Address space %s is not allowed in this arena\n", shd_get_address_space_name(ptr_type.address_space));
        shd_error_die();
    }
    if (!ptr_type.is_reference && !arena->config.target.memory.address_spaces[ptr_type.address_space].physical) {
        shd_error_print("Address space %s is not physical in this arena\n", shd_get_address_space_name(ptr_type.address_space));
        shd_error_die();
    }
    assert(ptr_type.pointed_type && "Shady does not support untyped pointers, but can infer them, see infer.c");
    if (ptr_type.pointed_type) {
        const Node* maybe_record_type = ptr_type.pointed_type;
        if (maybe_record_type->tag == NominalType_TAG)
            maybe_record_type = shd_get_nominal_type_body(maybe_record_type);
        if (maybe_record_type && maybe_record_type->tag == RecordType_TAG && maybe_record_type->payload.record_type.special == DecorateBlock) {
            return NULL;
        }
        switch (ptr_type.pointed_type->tag) {
            // these are unconditionally OK
            case FnType_TAG:
                return NULL;
            case ArrType_TAG:
                return NULL; // allows pointer to unsized arrays
            case RecordType_TAG:
                switch (ptr_type.pointed_type->payload.record_type.special) {
                    case NotSpecial: break;
                    case DecorateBlock: return NULL; // explicitly OK even if the pointee is not a datatype
                    case MultipleReturn: shd_error("Pointers to multiple-return definitions are not allowed")
                }
                break;
            default:
                break;
        }
        if (!shd_is_data_type(ptr_type.pointed_type))
            shd_error("Found an illegal pointer");
    }
    return NULL;
}

const Type* _shd_check_type_param(IrArena* arena, Param variable) {
    assert(shd_is_value_type(variable.type));
    return variable.type;
}

const Type* _shd_check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    shd_error("should never happen");
}

const Type* _shd_check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .scope = shd_get_arena_config(arena)->target.scopes.constants,
        .type = int_type(arena, (Int) { .width = lit.width, .is_signed = lit.is_signed })
    });
}

const Type* _shd_check_type_float_literal(IrArena* arena, FloatLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
            .scope = shd_get_arena_config(arena)->target.scopes.constants,
        .type = float_type(arena, (Float) { .width = lit.width })
    });
}

const Type* _shd_check_type_true_lit(IrArena* arena) {
    return qualified_type(arena, (QualifiedType) {
        .type = bool_type(arena),
        .scope = shd_get_arena_config(arena)->target.scopes.constants,
    });
}

const Type* _shd_check_type_false_lit(IrArena* arena) {
    return qualified_type(arena, (QualifiedType) {
        .type = bool_type(arena),
        .scope = shd_get_arena_config(arena)->target.scopes.constants,
    });
}

const Type* _shd_check_type_string_lit(IrArena* arena, StringLiteral str_lit) {
    const Type* t = arr_type(arena, (ArrType) {
        .element_type = shd_int8_type(arena),
        .size = shd_int32_literal(arena, strlen(str_lit.string))
    });
    return qualified_type(arena, (QualifiedType) {
        .type = t,
        .scope = shd_get_arena_config(arena)->target.scopes.constants,
    });
}

const Type* _shd_check_type_null_ptr(IrArena* a, NullPtr payload) {
    assert(shd_is_data_type(payload.ptr_type) && payload.ptr_type->tag == PtrType_TAG);
    return qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.constants, payload.ptr_type);
}

const Type* _shd_check_type_composite(IrArena* arena, Composite composite) {
    if (composite.type) {
        assert(shd_is_data_type(composite.type));
        Nodes expected_member_types = shd_get_composite_type_element_types(composite.type);
        ShdScope scope = shd_get_arena_config(arena)->target.scopes.constants;
        assert(composite.contents.count == expected_member_types.count);
        for (size_t i = 0; i < composite.contents.count; i++) {
            const Type* element_type = composite.contents.nodes[i]->type;
            scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&element_type));
            assert(shd_is_subtype(expected_member_types.nodes[i], element_type));
        }
        return qualified_type(arena, (QualifiedType) {
            .scope = scope,
            .type = composite.type
        });
    }
    ShdScope scope = shd_get_arena_config(arena)->target.scopes.constants;
    LARRAY(const Type*, member_ts, composite.contents.count);
    for (size_t i = 0; i < composite.contents.count; i++) {
        const Type* element_type = composite.contents.nodes[i]->type;
        scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&element_type));
        member_ts[i] = element_type;
    }
    return qualified_type(arena, (QualifiedType) {
        .scope = scope,
        .type = record_type(arena, (RecordType) {
            .members = shd_nodes(arena, composite.contents.count, member_ts)
        })
    });
}

const Type* _shd_check_type_extract(IrArena* a, Extract extract) {
    const Type* t = extract.composite->type;
    ShdScope scope;
    if (t->tag == RecordType_TAG && t->payload.record_type.special == MultipleReturn) {
        t = t->payload.record_type.members.nodes[shd_get_int_value(extract.selector, false)];
        scope = shd_deconstruct_qualified_type(&t);
    } else {
        scope = shd_deconstruct_qualified_type(&t);
        shd_enter_composite_type(&t, &scope, extract.selector, true);
    }

    return qualified_type_helper(a, scope, t);
}

const Type* _shd_check_type_insert(IrArena* a, Insert insert) {
    const Type* t = insert.composite->type;
    ShdScope scope;

    scope = shd_deconstruct_qualified_type(&t);
    shd_enter_composite_type(&t, &scope, insert.selector, true);

    const Type* inserted_data_type = insert.replacement->type;
    scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&inserted_data_type));
    assert(shd_is_subtype(t, inserted_data_type) && "inserting data into a composite, but it doesn't match the target and indices");
    return qualified_type(a, (QualifiedType) {
        .scope = scope,
        .type = shd_get_unqualified_type(insert.composite->type),
    });
}

const Type* _shd_check_type_fill(IrArena* arena, Fill payload) {
    assert(shd_is_data_type(payload.type));
    const Node* element_t = shd_get_fill_type_element_type(payload.type);
    const Node* value_t = payload.value->type;
    ShdScope s = shd_deconstruct_qualified_type(&value_t);
    assert(shd_is_subtype(element_t, value_t));
    return qualified_type(arena, (QualifiedType) {
        .scope = s,
        .type = payload.type
    });
}

const Type* _shd_check_type_undef(IrArena* arena, Undef payload) {
    assert(shd_is_data_type(payload.type));
    return qualified_type(arena, (QualifiedType) {
        .scope = shd_get_arena_config(arena)->target.scopes.bottom,
        .type = payload.type
    });
}

const Type* _shd_check_type_mem_and_value(IrArena* arena, MemAndValue mav) {
    return mav.value->type;
}

const Type* _shd_check_type_fn_addr(IrArena* arena, FnAddr fn_addr) {
    assert(fn_addr.fn->type->tag == FnType_TAG);
    assert(fn_addr.fn->tag == Function_TAG);
    return qualified_type(arena, (QualifiedType) {
        .scope = shd_get_arena_config(arena)->target.scopes.constants,
        .type = ptr_type(arena, (PtrType) {
            .pointed_type = fn_addr.fn->type,
            .address_space = AsCode /* the actual AS does not matter because these are opaque anyways */,
            .is_reference = !shd_get_arena_config(arena)->target.memory.address_spaces[AsCode].physical,
        })
    });
}

const Type* _shd_check_type_prim_op(IrArena* arena, PrimOp prim_op) {
    for (size_t i = 0; i < prim_op.operands.count; i++) {
        const Node* operand = prim_op.operands.nodes[i];
        assert(operand && is_value(operand));
    }

    bool extended = false;
    bool ordered = false;
    switch (prim_op.op) {
        case neg_op: {
            assert(prim_op.operands.count == 1);

            const Type* type = shd_first(prim_op.operands)->type;
            assert(shd_is_arithm_type(shd_get_maybe_vector_type_element(shd_get_unqualified_type(type))));
            return type;
        }
        case rshift_arithm_op:
        case rshift_logical_op:
        case lshift_op: {
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = shd_first(prim_op.operands)->type;
            const Type* second_operand_type = prim_op.operands.nodes[1]->type;

            ShdScope result_scope = shd_deconstruct_qualified_type(&first_operand_type);
            result_scope = shd_combine_scopes(result_scope, shd_deconstruct_qualified_type(&second_operand_type));

            size_t value_simd_width = shd_deconstruct_maybe_vector_type(&first_operand_type);
            size_t shift_simd_width = shd_deconstruct_maybe_vector_type(&second_operand_type);
            assert(value_simd_width == shift_simd_width);

            assert(first_operand_type->tag == Int_TAG);
            assert(second_operand_type->tag == Int_TAG);

            return qualified_type_helper(arena, result_scope, shd_maybe_vector_type_helper(first_operand_type, value_simd_width));
        }
        case add_carry_op:
        case sub_borrow_op:
        case mul_extended_op: extended = true; SHADY_FALLTHROUGH;
        case min_op:
        case max_op:
        case add_op:
        case sub_op:
        case mul_op:
        case div_op:
        case mod_op: {
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = shd_get_unqualified_type(shd_first(prim_op.operands)->type);

            ShdScope result_scope = shd_get_arena_config(arena)->target.scopes.constants;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                ShdScope operand_scope = shd_deconstruct_qualified_type(&operand_type);

                assert(shd_is_arithm_type(shd_get_maybe_vector_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_scope = shd_combine_scopes(result_scope, operand_scope);
            }

            const Type* result_t = first_operand_type;
            if (extended) {
                // TODO: assert unsigned
                result_t = record_type(arena, (RecordType) {.members = mk_nodes(arena, result_t, result_t)});
            }
            return qualified_type_helper(arena, result_scope, result_t);
        }

        case not_op: {
            assert(prim_op.operands.count == 1);

            const Type* type = shd_first(prim_op.operands)->type;
            assert(shd_has_boolean_ops(shd_get_maybe_vector_type_element(shd_get_unqualified_type(type))));
            return type;
        }
        case or_op:
        case xor_op:
        case and_op: {
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = shd_get_unqualified_type(shd_first(prim_op.operands)->type);

            ShdScope result_scope = shd_get_arena_config(arena)->target.scopes.constants;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                ShdScope operand_scope = shd_deconstruct_qualified_type(&operand_type);

                assert(shd_has_boolean_ops(shd_get_maybe_vector_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_scope = shd_combine_scopes(result_scope, operand_scope);
            }

            return qualified_type_helper(arena, result_scope, first_operand_type);
        }
        case lt_op:
        case lte_op:
        case gt_op:
        case gte_op: ordered = true; SHADY_FALLTHROUGH
        case eq_op:
        case neq_op: {
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = shd_get_unqualified_type(shd_first(prim_op.operands)->type);
            size_t first_operand_width = shd_get_maybe_vector_type_width(first_operand_type);

            ShdScope result_scope = shd_get_arena_config(arena)->target.scopes.constants;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                ShdScope operand_scope = shd_deconstruct_qualified_type(&operand_type);

                assert((ordered ? shd_is_ordered_type : shd_is_comparable_type)(shd_get_maybe_vector_type_element(operand_type)));
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_scope = shd_combine_scopes(result_scope, operand_scope);
            }

            return qualified_type_helper(arena, result_scope, shd_maybe_vector_type_helper(bool_type(arena), first_operand_width));
        }
        case sqrt_op:
        case inv_sqrt_op:
        case floor_op:
        case ceil_op:
        case round_op:
        case fract_op:
        case sin_op:
        case cos_op:
        case exp_op:
        {
            assert(prim_op.operands.count == 1);
            const Node* src_type = shd_first(prim_op.operands)->type;
            ShdScope scope = shd_deconstruct_qualified_type(&src_type);
            size_t width = shd_deconstruct_maybe_vector_type(&src_type);
            assert(src_type->tag == Float_TAG);
            return qualified_type_helper(arena, scope, shd_maybe_vector_type_helper(src_type, width));
        }
        case pow_op: {
            assert(prim_op.operands.count == 2);
            const Type* first_operand_type = shd_get_unqualified_type(shd_first(prim_op.operands)->type);

            ShdScope result_scope = shd_get_arena_config(arena)->target.scopes.constants;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                ShdScope operand_scope = shd_deconstruct_qualified_type(&operand_type);

                assert(shd_get_maybe_vector_type_element(operand_type)->tag == Float_TAG);
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_scope = shd_combine_scopes(result_scope, operand_scope);
            }

            return qualified_type_helper(arena, result_scope, first_operand_type);
        }
        case fma_op: {
            assert(prim_op.operands.count == 3);
            const Type* first_operand_type = shd_get_unqualified_type(shd_first(prim_op.operands)->type);

            ShdScope result_scope = shd_get_arena_config(arena)->target.scopes.constants;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                const Type* operand_type = arg->type;
                ShdScope operand_scope = shd_deconstruct_qualified_type(&operand_type);

                assert(shd_get_maybe_vector_type_element(operand_type)->tag == Float_TAG);
                assert(first_operand_type == operand_type &&  "operand type mismatch");

                result_scope = shd_combine_scopes(result_scope, operand_scope);
            }

            return qualified_type_helper(arena, result_scope, first_operand_type);
        }
        case abs_op:
        case sign_op:
        {
            assert(prim_op.operands.count == 1);
            const Node* src_type = shd_first(prim_op.operands)->type;
            ShdScope scope = shd_deconstruct_qualified_type(&src_type);
            size_t width = shd_deconstruct_maybe_vector_type(&src_type);
            assert(src_type->tag == Float_TAG || (src_type->tag == Int_TAG && src_type->payload.int_type.is_signed));
            return qualified_type_helper(arena, scope, shd_maybe_vector_type_helper(src_type, width));
        }
        case select_op: {
            assert(prim_op.operands.count == 3);
            const Type* condition_type = prim_op.operands.nodes[0]->type;
            ShdScope scope = shd_deconstruct_qualified_type(&condition_type);
            size_t width = shd_deconstruct_maybe_vector_type(&condition_type);

            const Type* alternatives_types[2];
            for (size_t i = 0; i < 2; i++) {
                alternatives_types[i] = prim_op.operands.nodes[1 + i]->type;
                scope = shd_combine_scopes(scope, shd_deconstruct_qualified_type(&alternatives_types[i]));
                size_t alternative_width = shd_deconstruct_maybe_vector_type(&alternatives_types[i]);
                assert(alternative_width == width);
            }

            assert(shd_is_subtype(bool_type(arena), condition_type));
            // todo find true supertype
            assert(are_types_identical(2, alternatives_types));

            return qualified_type_helper(arena, scope, shd_maybe_vector_type_helper(alternatives_types[0], width));
        }
        case shuffle_op: {
            assert(prim_op.operands.count >= 2);
            const Node* lhs = prim_op.operands.nodes[0];
            const Node* rhs = prim_op.operands.nodes[1];
            const Type* lhs_t = lhs->type;
            const Type* rhs_t = rhs->type;
            ShdScope lhs_scope = shd_deconstruct_qualified_type(&lhs_t);
            ShdScope rhs_scope = shd_deconstruct_qualified_type(&rhs_t);
            assert(lhs_t->tag == VectorType_TAG && rhs_t->tag == VectorType_TAG);
            int64_t total_size = lhs_t->payload.vector_type.width + rhs_t->payload.vector_type.width;
            const Type* element_t = lhs_t->payload.vector_type.element_type;
            assert(element_t == rhs_t->payload.vector_type.element_type);

            size_t indices_count = prim_op.operands.count - 2;
            const Node** indices = &prim_op.operands.nodes[2];
            ShdScope scope = shd_combine_scopes(lhs_scope, rhs_scope);
            for (size_t i = 0; i < indices_count; i++) {
                scope = shd_combine_scopes(scope, shd_get_qualified_type_scope(indices[i]->type));
                int64_t index = shd_get_int_literal_value(*shd_resolve_to_int_literal(indices[i]), true);
                assert(index < 0 /* poison */ || (index >= 0 && index < total_size && "shuffle element out of range"));
            }
            return qualified_type_helper(arena, scope, vector_type(arena, (VectorType) {.element_type = element_t, .width = indices_count}));
        }
        // Mask management
        case empty_mask_op: {
            assert(prim_op.operands.count == 0);
            return qualified_type_helper(arena, shd_get_arena_config(arena)->target.scopes.constants, shd_get_exec_mask_type(arena));
        }
        case mask_is_thread_active_op: {
            assert(prim_op.operands.count == 2);
            return qualified_type(arena, (QualifiedType) {
                .scope = shd_combine_scopes(shd_get_qualified_type_scope(prim_op.operands.nodes[0]->type), shd_get_qualified_type_scope(prim_op.operands.nodes[1]->type)),
                .type = bool_type(arena)
            });
        }
        default: assert(false);
    }
}

const Type* _shd_check_type_size_of(IrArena* a, SizeOf payload) {
    return qualified_type(a, (QualifiedType) {
        .scope = shd_get_arena_config(a)->target.scopes.constants,
        .type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false })
    });
}

const Type* _shd_check_type_align_of(IrArena* a, AlignOf payload) {
    return qualified_type(a, (QualifiedType) {
        .scope = shd_get_arena_config(a)->target.scopes.constants,
        .type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false })
    });
}

const Type* _shd_check_type_offset_of(IrArena* a, OffsetOf payload) {
    const Type* optype = payload.idx->type;
    ShdScope index_scope = shd_deconstruct_qualified_type(&optype);
    assert(index_scope == shd_get_arena_config(a)->target.scopes.constants && optype->tag == Int_TAG);
    return qualified_type(a, (QualifiedType) {
        .scope = shd_get_arena_config(a)->target.scopes.constants,
        .type = int_type(a, (Int) { .width = a->config.target.memory.ptr_size, .is_signed = false })
    });
}

const Type* _shd_check_type_scope_cast(IrArena* a, ScopeCast cast) {
    const Type* operand_type = shd_get_unqualified_type(cast.src->type);
    return qualified_type(a, (QualifiedType) {
        .scope = cast.scope,
        .type = operand_type
    });
}

const Type* _shd_check_type_bit_cast(IrArena* a, BitCast cast) {
    const Type* src_type = cast.src->type;
    ShdScope src_scope = shd_deconstruct_qualified_type(&src_type);

    assert(shd_is_data_type(cast.type));
    assert(shd_is_bitcast_legal(src_type, cast.type));

    return qualified_type(a, (QualifiedType) {
        .scope = src_scope,
        .type = cast.type
    });
}

const Type* _shd_check_type_conversion(IrArena* a, Conversion conversion) {
    const Type* src_type = conversion.src->type;
    ShdScope src_scope = shd_deconstruct_qualified_type(&src_type);

    const Type* dst_type = conversion.type;
    assert(shd_is_data_type(dst_type));
    assert(shd_is_conversion_legal(src_type, dst_type));

    return qualified_type(a, (QualifiedType) {
        .scope = src_scope,
        .type = dst_type
    });
}

const Type* _shd_check_type_generic_ptr_cast(IrArena* a, GenericPtrCast generic_ptr_cast) {
    const Type* src_type = generic_ptr_cast.src->type;
    ShdScope src_scope = shd_deconstruct_qualified_type(&src_type);

    assert(src_type->tag == PtrType_TAG);
    PtrType payload = src_type->payload.ptr_type;
    if (payload.address_space == AsGeneric)
        shd_error("GenericPtrCast: source cannot be already a generic pointer.");

    payload.address_space = AsGeneric;

    const Type* dst_type = ptr_type(a, payload);
    return qualified_type(a, (QualifiedType) {
        .scope = src_scope,
        .type = dst_type
    });
}

const Type* _shd_check_type_ext_value(IrArena* arena, ExtValue payload) {
    return payload.result_t ? payload.result_t : unit_type(arena);
}

const Type* _shd_check_type_ext_instr(IrArena* arena, ExtInstr payload) {
    return payload.result_t ? payload.result_t : unit_type(arena);
}

const Type* _shd_check_type_ext_terminator(IrArena* arena, ExtTerminator payload) {
    return noret_type(arena);
}

static void check_arguments_types_against_parameters_helper(Nodes param_types, Nodes arg_types) {
    if (param_types.count != arg_types.count)
        shd_error("Mismatched number of arguments/parameters");
    for (size_t i = 0; i < param_types.count; i++) {
        assert(shd_is_value_type(param_types.nodes[i]));
        shd_check_subtype(param_types.nodes[i], arg_types.nodes[i]);
    }
}

/// Shared logic between indirect calls and tailcalls
static Nodes check_value_call(const Type* callee_type, Nodes argument_types) {
    assert(callee_type->tag == FnType_TAG);
    const FnType* fn_type = &callee_type->payload.fn_type;
    check_arguments_types_against_parameters_helper(fn_type->param_types, argument_types);
    // TODO force the return types to be varying if the callee is not uniform
    return fn_type->return_types;
}

const Type* _shd_check_type_call(IrArena* arena, Call call) {
    Nodes args = call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    Nodes argument_types = shd_get_values_types(arena, args);
    return shd_maybe_multiple_return(arena, check_value_call(call.callee->type, argument_types));
}

const Type* _shd_check_type_indirect_call(IrArena* arena, IndirectCall call) {
    assert(is_value(call.callee));
    const Type* callee_type = call.callee->type;
    SHADY_UNUSED bool callee_uniform = shd_deconstruct_qualified_type(&callee_type);
    AddressSpace as = shd_deconstruct_pointer_type(&callee_type);
    assert(as == AsCode);

    Nodes args = call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    Nodes argument_types = shd_get_values_types(arena, args);
    return shd_maybe_multiple_return(arena, check_value_call(callee_type, argument_types));
}

const Type* _shd_check_type_indirect_tail_call(IrArena* arena, IndirectTailCall tail_call) {
    assert(is_value(tail_call.callee));
    const Type* callee_type = tail_call.callee->type;
    shd_deconstruct_qualified_type(&callee_type);
    AddressSpace as = shd_deconstruct_pointer_type(&callee_type);
    assert(as == AsCode);

    Nodes args = tail_call.args;
    for (size_t i = 0; i < args.count; i++) {
        const Node* argument = args.nodes[i];
        assert(is_value(argument));
    }
    check_value_call(callee_type, shd_get_values_types(arena, tail_call.args));
    // TODO: check it matches function ?
    // assert(check_value_call(callee_type, shd_get_values_types(arena, tail_call.args)).count == 0);
    return noret_type(arena);
}

static void ensure_types_are_data_types(const Nodes* yield_types) {
    for (size_t i = 0; i < yield_types->count; i++) {
        assert(shd_is_data_type(yield_types->nodes[i]));
    }
}

const Type* _shd_check_type_if_instr(IrArena* arena, If if_instr) {
    assert(if_instr.tail && is_abstraction(if_instr.tail));
    ensure_types_are_data_types(&if_instr.yield_types);
    if (shd_get_unqualified_type(if_instr.condition->type) != bool_type(arena))
        shd_error("condition of an if should be bool");
    // TODO check the contained Merge instrs
    if (if_instr.yield_types.count > 0)
        assert(if_instr.if_false);

    check_arguments_types_against_parameters_helper(shd_get_param_types(arena, get_abstraction_params(if_instr.tail)), shd_add_qualifiers(arena, if_instr.yield_types, shd_get_arena_config(arena)->target.scopes.bottom));
    return noret_type(arena);
}

const Type* _shd_check_type_match_instr(IrArena* arena, Match match_instr) {
    ensure_types_are_data_types(&match_instr.yield_types);
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return noret_type(arena);
}

const Type* _shd_check_type_loop_instr(IrArena* arena, Loop loop_instr) {
    ensure_types_are_data_types(&loop_instr.yield_types);
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return noret_type(arena);
}

const Type* _shd_check_type_control(IrArena* arena, Control control) {
    ensure_types_are_data_types(&control.yield_types);
    // TODO check it then !
    const Node* join_point = shd_first(get_abstraction_params(control.inside));

    const Type* join_point_type = join_point->type;
    shd_deconstruct_qualified_type(&join_point_type);
    assert(join_point_type->tag == JoinPointType_TAG);

    Nodes join_point_yield_types = join_point_type->payload.join_point_type.yield_types;
    assert(join_point_yield_types.count == control.yield_types.count);
    for (size_t i = 0; i < control.yield_types.count; i++) {
        assert(shd_is_subtype(control.yield_types.nodes[i], join_point_yield_types.nodes[i]));
    }

    assert(get_abstraction_params(control.tail).count == control.yield_types.count);

    return noret_type(arena);
}

const Type* _shd_check_type_comment(IrArena* arena, SHADY_UNUSED Comment payload) {
    return empty_multiple_return_type(arena);
}

const Type* _shd_check_type_stack_alloc(IrArena* a, StackAlloc alloc) {
    assert(is_type(alloc.type));
    return qualified_type(a, (QualifiedType) {
        .scope = shd_get_addr_space_scope(AsPrivate),
        .type = ptr_type(a, (PtrType) {
            .pointed_type = alloc.type,
            .address_space = AsPrivate,
            .is_reference = false
        })
    });
}

const Type* _shd_check_type_local_alloc(IrArena* a, LocalAlloc alloc) {
    assert(is_type(alloc.type));
    return qualified_type(a, (QualifiedType) {
        .scope = shd_get_addr_space_scope(AsFunction),
        .type = ptr_type(a, (PtrType) {
            .pointed_type = alloc.type,
            .address_space = AsFunction,
            .is_reference = true
        })
    });
}

const Type* _shd_check_type_load(IrArena* a, Load load) {
    const Node* ptr_type = load.ptr->type;
    ShdScope ptr_scope = shd_deconstruct_qualified_type(&ptr_type);
    size_t width = shd_deconstruct_maybe_vector_type(&ptr_type);

    assert(ptr_type->tag == PtrType_TAG);
    const PtrType* node_ptr_type_ = &ptr_type->payload.ptr_type;
    const Type* elem_type = node_ptr_type_->pointed_type;
    elem_type = shd_maybe_vector_type_helper(elem_type, width);
    return qualified_type_helper(a, shd_combine_scopes(ptr_scope, shd_get_addr_space_scope(ptr_type->payload.ptr_type.address_space)), elem_type);
}

const Type* _shd_check_type_store(IrArena* a, Store store) {
    const Node* ptr_type = store.ptr->type;
    shd_deconstruct_qualified_type(&ptr_type);
    size_t width = shd_deconstruct_maybe_vector_type(&ptr_type);
    assert(ptr_type->tag == PtrType_TAG);
    const PtrType* ptr_type_payload = &ptr_type->payload.ptr_type;
    const Type* elem_type = ptr_type_payload->pointed_type;
    assert(elem_type);
    elem_type = shd_maybe_vector_type_helper(elem_type, width);
    const Type* expected_stored_type = qualified_type(a, (QualifiedType) {
        .scope = shd_get_arena_config(a)->target.scopes.bottom,
        .type = elem_type
    });

    assert(shd_is_subtype(expected_stored_type, store.value->type));
    return empty_multiple_return_type(a);
}

const Type* _shd_check_type_ptr_array_element_offset(IrArena* a, PtrArrayElementOffset lea) {
    const Type* base_ptr_type = lea.ptr->type;
    ShdScope ptr_scope = shd_deconstruct_qualified_type(&base_ptr_type);
    assert(base_ptr_type->tag == PtrType_TAG && "lea expects a ptr or ref as a base");
    const Type* pointee_type = base_ptr_type->payload.ptr_type.pointed_type;

    assert(lea.offset);
    const Type* offset_type = lea.offset->type;
    ShdScope offset_scope = shd_deconstruct_qualified_type(&offset_type);
    assert(offset_type->tag == Int_TAG && "lea expects an integer offset");

    const IntLiteral* lit = shd_resolve_to_int_literal(lea.offset);
    bool offset_is_zero = lit && lit->value == 0;
    assert((offset_is_zero || !base_ptr_type->payload.ptr_type.is_reference) && "if an offset is used, the base cannot be a reference");
    assert((offset_is_zero || shd_is_data_type(pointee_type)) && "if an offset is used, the base must point to a data type");

    return qualified_type(a, (QualifiedType) {
        .scope = shd_combine_scopes(ptr_scope, offset_scope),
        .type = ptr_type(a, (PtrType) {
            .pointed_type = pointee_type,
            .address_space = base_ptr_type->payload.ptr_type.address_space,
            .is_reference = base_ptr_type->payload.ptr_type.is_reference
        })
    });
}

const Type* _shd_check_type_ptr_composite_element(IrArena* a, PtrCompositeElement lea) {
    const Type* base_ptr_type = lea.ptr->type;
    ShdScope s = shd_deconstruct_qualified_type(&base_ptr_type);
    assert(base_ptr_type->tag == PtrType_TAG && "lea expects a ptr or ref as a base");
    const Type* pointee_type = base_ptr_type->payload.ptr_type.pointed_type;

    shd_enter_composite_type(&pointee_type, &s, lea.index, true);

    return qualified_type(a, (QualifiedType) {
        .scope = s,
        .type = ptr_type(a, (PtrType) {
            .pointed_type = pointee_type,
            .address_space = base_ptr_type->payload.ptr_type.address_space,
            .is_reference = base_ptr_type->payload.ptr_type.is_reference
        })
    });
}

const Type* _shd_check_type_copy_bytes(IrArena* a, CopyBytes copy_bytes) {
    const Type* dst_t = copy_bytes.dst->type;
    shd_deconstruct_qualified_type(&dst_t);
    assert(dst_t->tag == PtrType_TAG);
    const Type* src_t = copy_bytes.src->type;
    shd_deconstruct_qualified_type(&src_t);
    assert(src_t);
    const Type* cnt_t = copy_bytes.count->type;
    shd_deconstruct_qualified_type(&cnt_t);
    assert(cnt_t->tag == Int_TAG);
    return empty_multiple_return_type(a);
}

const Type* _shd_check_type_fill_bytes(IrArena* a, FillBytes fill_bytes) {
    const Type* dst_t = fill_bytes.dst->type;
    shd_deconstruct_qualified_type(&dst_t);
    assert(dst_t->tag == PtrType_TAG);
    const Type* src_t = fill_bytes.src->type;
    shd_deconstruct_qualified_type(&src_t);
    assert(src_t && src_t->tag == Int_TAG && src_t->payload.int_type.width <= a->config.target.memory.word_size);
    const Type* cnt_t = fill_bytes.count->type;
    shd_deconstruct_qualified_type(&cnt_t);
    assert(cnt_t->tag == Int_TAG);
    return empty_multiple_return_type(a);
}

const Type* _shd_check_type_push_stack(IrArena* a, PushStack payload) {
    assert(payload.value);
    return empty_multiple_return_type(a);
}

const Type* _shd_check_type_pop_stack(IrArena* a, PopStack payload) {
    return qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, payload.type);
}

const Type* _shd_check_type_set_stack_size(IrArena* a, SetStackSize payload) {
    assert(shd_get_unqualified_type(payload.value->type) == shd_uint32_type(a));
    return empty_multiple_return_type(a);
}

const Type* _shd_check_type_get_stack_size(IrArena* a, SHADY_UNUSED GetStackSize ss) {
    return qualified_type(a, (QualifiedType) { .scope = shd_get_arena_config(a)->target.scopes.bottom, .type = shd_uint32_type(a) });
}

const Type* _shd_check_type_get_stack_base_addr(IrArena* a, SHADY_UNUSED GetStackBaseAddr gsba) {
    const Node* ptr = ptr_type(a, (PtrType) { .pointed_type = shd_uint8_type(a), .address_space = AsPrivate});
    return qualified_type(a, (QualifiedType) { .scope = shd_get_arena_config(a)->target.scopes.bottom, .type = ptr });
}

const Type* _shd_check_type_debug_printf(IrArena* a, DebugPrintf payload) {
    return empty_multiple_return_type(a);
}

static void check_basic_block_call(const Node* block, Nodes argument_types) {
    assert(is_basic_block(block));
    assert(block->type->tag == BBType_TAG);
    BBType bb_type = block->type->payload.bb_type;
    check_arguments_types_against_parameters_helper(bb_type.param_types, argument_types);
}

const Type* _shd_check_type_jump(IrArena* arena, Jump jump) {
    for (size_t i = 0; i < jump.args.count; i++) {
        const Node* argument = jump.args.nodes[i];
        assert(is_value(argument));
    }

    check_basic_block_call(jump.target, shd_get_values_types(arena, jump.args));
    return noret_type(arena);
}

const Type* _shd_check_type_branch(IrArena* arena, Branch payload) {
    assert(payload.true_jump->tag == Jump_TAG);
    assert(payload.false_jump->tag == Jump_TAG);
    return noret_type(arena);
}

const Type* _shd_check_type_br_switch(IrArena* arena, Switch payload) {
    for (size_t i = 0; i < payload.case_jumps.count; i++)
        assert(payload.case_jumps.nodes[i]->tag == Jump_TAG);
    assert(payload.case_values.count == payload.case_jumps.count);
    assert(payload.default_jump->tag == Jump_TAG);
    return noret_type(arena);
}

const Type* _shd_check_type_join(IrArena* arena, Join join) {
    for (size_t i = 0; i < join.args.count; i++) {
        const Node* argument = join.args.nodes[i];
        assert(is_value(argument));
    }

    const Type* join_target_type = join.join_point->type;

    shd_deconstruct_qualified_type(&join_target_type);
    assert(join_target_type->tag == JoinPointType_TAG);

    Nodes join_point_param_types = join_target_type->payload.join_point_type.yield_types;
    join_point_param_types = shd_add_qualifiers(arena, join_point_param_types, shd_get_arena_config(arena)->target.scopes.bottom);

    check_arguments_types_against_parameters_helper(join_point_param_types, shd_get_values_types(arena, join.args));

    return noret_type(arena);
}

const Type* _shd_check_type_unreachable(IrArena* arena, SHADY_UNUSED Unreachable u) {
    return noret_type(arena);
}

const Type* _shd_check_type_merge_continue(IrArena* arena, MergeContinue mc) {
    // TODO check it
    return noret_type(arena);
}

const Type* _shd_check_type_merge_break(IrArena* arena, MergeBreak mc) {
    // TODO check it
    return noret_type(arena);
}

const Type* _shd_check_type_merge_selection(IrArena* arena, SHADY_UNUSED MergeSelection payload) {
    // TODO check it
    return noret_type(arena);
}

const Type* _shd_check_type_fn_ret(IrArena* arena, Return ret) {
    // assert(ret.fn);
    // TODO check it then !
    return noret_type(arena);
}

const Type* _shd_check_type_fn_type(IrArena* arena, FnType fn) {
    for (size_t i = 0; i < fn.return_types.count; i++) {
        assert(shd_is_value_type(fn.return_types.nodes[i]));
    }
    return NULL;
}

const Type* _shd_check_type_fun(IrArena* arena, Function fn) {
    for (size_t i = 0; i < fn.return_types.count; i++) {
        assert(shd_is_value_type(fn.return_types.nodes[i]));
    }
    return fn_type(arena, (FnType) { .param_types = shd_get_param_types(arena, (&fn)->params), .return_types = (&fn)->return_types });
}

const Type* _shd_check_type_basic_block(IrArena* arena, BasicBlock bb) {
    return bb_type(arena, (BBType) { .param_types = shd_get_param_types(arena, (&bb)->params) });
}

const Type* _shd_check_type_global_variable(IrArena* arena, GlobalVariable global_variable) {
    assert(is_type(global_variable.type));

    assert(global_variable.address_space < NumAddressSpaces);

    return qualified_type_helper(arena, shd_get_arena_config(arena)->target.scopes.constants, ptr_type(arena, (PtrType) {
        .pointed_type = global_variable.type,
        .address_space = global_variable.address_space,
        .is_reference = global_variable.is_ref,
    }));
}

const Type* _shd_check_type_builtin_ref(IrArena* arena, BuiltinRef ref) {
    ShdScope scope = shd_get_builtin_scope(ref.builtin);
    return qualified_type_helper(arena, scope, ptr_type(arena, (PtrType) {
        .pointed_type = shd_get_builtin_type(arena, ref.builtin),
        .address_space = shd_get_builtin_address_space(ref.builtin),
        .is_reference = true,
    }));
}

const Type* _shd_check_type_constant(IrArena* arena, Constant cnst) {
    assert(shd_is_data_type(cnst.type_hint));
    return qualified_type_helper(arena, shd_get_arena_config(arena)->target.scopes.constants, cnst.type_hint);
}

#include "type_generated.c"

#pragma GCC diagnostic pop
