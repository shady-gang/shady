#include "type.h"

#include "log.h"
#include "arena.h"
#include "portability.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

#define empty() nodes(arena, 0, NULL)
#define singleton(t) singleton_impl(arena, t)
Nodes singleton_impl(IrArena* arena, const Type* type) {
    const Type* arr[] = { type };
    return nodes(arena, 1, arr);
}

bool is_type(const Node* node) {
    switch (node->tag) {
#define NODEDEF(_, _2, _3, name, _4) case name##_TAG:
TYPE_NODES()
#undef NODEDEF
                 return true;
        default: return false;
    }
}

bool is_subtype(const Type* supertype, const Type* type) {
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
        case QualifiedType_TAG: {
            // uniform T <: varying T
            if (supertype->payload.qualified_type.is_uniform && !type->payload.qualified_type.is_uniform)
                return false;
            return is_subtype(supertype->payload.qualified_type.type, type->payload.qualified_type.type);
        }
        case RecordType_TAG: {
            const Nodes* supermembers = &supertype->payload.record_type.members;
            const Nodes* members = &type->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->nodes[i], members->nodes[i]))
                    return false;
            }
            return true;
        }
        case FnType_TAG:
            if (supertype->payload.fn.is_basic_block != type->payload.fn.is_basic_block)
                return false;
            // check returns
            if (supertype->payload.fn.return_types.count != type->payload.fn.return_types.count)
                return false;
            for (size_t i = 0; i < type->payload.fn.return_types.count; i++)
                if (!is_subtype(supertype->payload.fn.return_types.nodes[i], type->payload.fn.return_types.nodes[i]))
                    return false;
            // check params
            const Nodes* superparams = &supertype->payload.fn_type.param_types;
            const Nodes* params = &type->payload.fn_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        case PtrType_TAG: {
            if (supertype->payload.ptr_type.address_space != type->payload.ptr_type.address_space)
                return false;
            return is_subtype(supertype->payload.ptr_type.pointed_type, type->payload.ptr_type.pointed_type);
        }
        case Int_TAG: return supertype->payload.int_type.width == type->payload.int_type.width;
        // simple types without a payload
        default: return true;
    }
    SHADY_UNREACHABLE;
}

void check_subtype(const Type* supertype, const Type* type) {
    if (!is_subtype(supertype, type)) {
        print_node(type);
        printf(" isn't a subtype of ");
        print_node(supertype);
        printf("\n");
        error("failed check_subtype")
    }
}

/// @deprecated
const Type* strip_qualifier(const Type* type, DivergenceQualifier* qual_out) {
    if (type->tag == QualifiedType_TAG) {
        *qual_out = type->payload.qualified_type.is_uniform ? Uniform : Varying;
        return type->payload.qualified_type.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

/// @deprecated
DivergenceQualifier get_qualifier(const Type* type) {
    DivergenceQualifier result;
    strip_qualifier(type, &result);
    return result;
}

const Type* without_qualifier(const Type* type) {
    DivergenceQualifier dontcare;
    return strip_qualifier(type, &dontcare);
}

Nodes extract_variable_types(IrArena* arena, const Nodes* variables) {
    LARRAY(const Type*, arr, variables->count);
    for (size_t i = 0; i < variables->count; i++)
        arr[i] = variables->nodes[i]->payload.var.type;
    return nodes(arena, variables->count, arr);
}

Nodes extract_types(IrArena* arena, Nodes values) {
    LARRAY(const Type*, arr, values.count);
    for (size_t i = 0; i < values.count; i++)
        arr[i] = values.nodes[i]->type;
    return nodes(arena, values.count, arr);
}

const Type* derive_fn_type(IrArena* arena, const Function* fn) {
    return fn_type(arena, (FnType) { .is_basic_block = fn->is_basic_block, .param_types = extract_variable_types(arena, &fn->params), .return_types = fn->return_types });
}

const Type* check_type_fn(IrArena* arena, Function fn) {
    assert(!fn.is_basic_block || fn.return_types.count == 0);
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = derive_fn_type(arena, &fn)
    });
}

const Type* check_type_global_variable(IrArena* arena, GlobalVariable global_variable) {
    return qualified_type(arena, (QualifiedType) {
        .type = ptr_type(arena, (PtrType) {
            .pointed_type = global_variable.type,
            .address_space = global_variable.address_space
        }),
        .is_uniform = true
    });
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
const Type* check_type_var(IrArena* arena, Variable variable) {
    assert(variable.type);
    assert(get_qualifier(variable.type) != Unknown);
    return variable.type;
}

const Type* check_type_qualified_type(IrArena* arena, QualifiedType qualified_type) {
    assert(get_qualifier(qualified_type.type) == Unknown);
    return NULL;
}

const Type* check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

const Type* check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = int_type(arena, (Int) { .width = lit.width })
    });
}

const Type* check_type_true_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }
const Type* check_type_false_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }

const Type* check_type_tuple(IrArena* arena, Tuple tuple) {
    return record_type(arena, (RecordType) {
        .members = extract_types(arena, tuple.contents),
        .must_be_deconstructed = false,
        .names = strings(arena, 0, NULL)
    });
}

const Type* check_type_fn_ret(IrArena* arena, Return ret) {
    // TODO check it then !
    return NULL;
}

const Type* wrap_multiple_yield_types(IrArena* arena, Nodes types) {
    switch (types.count) {
        case 0: return unit_type(arena);
        case 1: return types.nodes[0];
        default: return record_type(arena, (RecordType) {
            .members = types,
            .names = strings(arena, 0, NULL),
            .must_be_deconstructed = true,
        });
    }
    SHADY_UNREACHABLE;
}

Nodes unwrap_multiple_yield_types(IrArena* arena, const Type* type) {
    switch (type->tag) {
        case Unit_TAG: return nodes(arena, 0, NULL);
        case RecordType_TAG:
            if (type->payload.record_type.must_be_deconstructed)
                return type->payload.record_type.members;
            // fallthrough
        default: return nodes(arena, 1, (const Node* []) { type });
    }
}

const Type* check_type_if_instr(IrArena* arena, If if_instr) {
    if (without_qualifier(if_instr.condition->type) != bool_type(arena))
        error("condition of a selection should be bool");
    // TODO check the contained Merge instrs
    if (if_instr.yield_types.count > 0)
        assert(if_instr.if_false);

    return wrap_multiple_yield_types(arena, if_instr.yield_types);
}

const Type* check_type_loop_instr(IrArena* arena, Loop loop_instr) {
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return wrap_multiple_yield_types(arena, loop_instr.yield_types);
}

const Type* check_type_match_instr(IrArena* arena, Match match_instr) {
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return wrap_multiple_yield_types(arena, match_instr.yield_types);
}

/// Oracle of what casts are legal
static bool is_reinterpret_cast_legal(const Type* src_type, const Type* dst_type) {
    // TODO implement rules
    assert(is_type(src_type) && is_type(dst_type));
    return true;
}

static const Type* get_actual_mask_type(IrArena* arena) {
    switch (arena->config.subgroup_mask_representation) {
        case SubgroupMaskAbstract: return mask_type(arena);
        case SubgroupMaskSpvKHRBallot: return pack_type(arena, (PackType) { .element_type = int32_type(arena), .width = 4 });
        default: error("unimplemented");
    }
}

/// Checks the operands to a Primop and returns the produced types
const Type* check_type_prim_op(IrArena* arena, PrimOp prim_op) {
    for (size_t i = 0; i < prim_op.operands.count; i++) {
        const Node* operand = prim_op.operands.nodes[i];
        assert(!operand || is_type(operand) || is_value(operand));
    }

    switch (prim_op.op) {
        case neg_op: {
            assert(prim_op.operands.count == 1);
            return prim_op.operands.nodes[0]->type;
            // return qualified_type(arena, (QualifiedType) { .is_uniform = , .type = bool_type(arena) });
        }
        case rshift_arithm_op:
        case rshift_logical_op:
        case lshift_op:

        case add_op:
        case sub_op:
        case mul_op:
        case div_op:
        case mod_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                DivergenceQualifier op_div;
                const Type* arg_actual_type = strip_qualifier(arg->type, &op_div);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform &= op_div == Uniform;
                // we work with numerical operands
                assert(arg_actual_type->tag == Int_TAG && "todo improve this check");
                assert(without_qualifier(prim_op.operands.nodes[0]->type)->tag == Int_TAG && "todo improve this check");
                assert(arg_actual_type->payload.int_type.width == without_qualifier(prim_op.operands.nodes[0]->type)->payload.int_type.width && "Arithmetic operations expect all operands to have the same widths");
            }

            IntSizes width = without_qualifier(prim_op.operands.nodes[0]->type)->payload.int_type.width;

            const Type* qt = qualified_type(arena, (QualifiedType) {
                .is_uniform = is_result_uniform,
                .type = int_type(arena, (Int) {
                    .width = width
                })
            });

            assert(qt->payload.qualified_type.type->payload.int_type.width == width);

            return qt;
        }

        case or_op:
        case xor_op:
        case and_op: {
            bool is_uniform = true;
            const Type* first_arg_type = without_qualifier(prim_op.operands.nodes[0]->type);
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                is_uniform &= get_qualifier(arg->type) == Uniform;
                const Type* arg_type = without_qualifier(arg->type);
                assert(arg_type == first_arg_type && "Operands must have the same type");
                switch (arg_type->tag) {
                    case Int_TAG:
                    case Bool_TAG: break;
                    default: error("Logical operations can only be applied on booleans and on integers");
                }
            }
            return qualified_type(arena, (QualifiedType) { .is_uniform = is_uniform, .type = first_arg_type });
        }

        case lt_op:
        case lte_op:
        case gt_op:
        case gte_op:
        case eq_op:
        case neq_op: {
            bool is_result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                DivergenceQualifier op_div;
                // TODO ensure these guys are compatible ?
                const Type* arg_actual_type = strip_qualifier(arg->type, &op_div);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform &= op_div == Uniform;
                assert(is_subtype(arg_actual_type, without_qualifier(prim_op.operands.nodes[0]->type)) && "Comparison operators need to be applied to the same types");
            }
            return qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = bool_type(arena) });
        }
        case push_stack_uniform_op:
        case push_stack_op: {
            assert(prim_op.operands.count == 2);
            const Type* element_type = prim_op.operands.nodes[0];
            assert(get_qualifier(element_type) == Unknown && "annotations do not go here");
            const Type* qual_element_type = qualified_type(arena, (QualifiedType) {
                .is_uniform = prim_op.op == push_stack_uniform_op,
                .type = element_type
            });
            // the operand has to be a subtype of the annotated type
            assert(is_subtype(qual_element_type, prim_op.operands.nodes[1]->type));
            return unit_type(arena);
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(prim_op.operands.count == 1);
            const Type* element_type = prim_op.operands.nodes[0];
            assert(get_qualifier(element_type) == Unknown && "annotations do not go here");
            return qualified_type(arena, (QualifiedType) { .is_uniform = prim_op.op == pop_stack_uniform_op, .type = element_type});
        }
        case load_op: {
            assert(prim_op.operands.count == 1);
            //const Type* elem_type = prim_op.operands.nodes[0];
            //assert(elem_type && is_type(elem_type));
            const Node* ptr = prim_op.operands.nodes[0];
            DivergenceQualifier qual;
            const Node* node_ptr_type = strip_qualifier(ptr->type, &qual);
            assert(qual != Unknown);
            assert(node_ptr_type->tag == PtrType_TAG);
            const PtrType* node_ptr_type_ = &node_ptr_type->payload.ptr_type;
            const Type* elem_type = node_ptr_type_->pointed_type;
            return qualified_type(arena, (QualifiedType) {
                .type = elem_type,
                .is_uniform = qual == Uniform
            });
        }
        case store_op: {
            assert(prim_op.operands.count == 2);
            //const Type* elem_type = prim_op.operands.nodes[0];
            //assert(elem_type && is_type(elem_type));
            const Node* ptr = prim_op.operands.nodes[0];
            DivergenceQualifier qual;
            const Node* node_ptr_type = strip_qualifier(ptr->type, &qual);
            assert(qual != Unknown);
            assert(node_ptr_type->tag == PtrType_TAG);
            const PtrType* node_ptr_type_ = &node_ptr_type->payload.ptr_type;
            const Type* elem_type = node_ptr_type_->pointed_type;
            // we don't enforce uniform stores - but we care about storing the right thing :)
            const Type* val_expected_type = qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = elem_type
            });

            const Node* val = prim_op.operands.nodes[1];
            assert(is_subtype(val_expected_type, val->type));
            return unit_type(arena);
        }
        case alloca_op: {
            assert(prim_op.operands.count == 1);
            const Type* elem_type = prim_op.operands.nodes[0];
            assert(is_type(elem_type));
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = ptr_type(arena, (PtrType) {
                    .pointed_type = elem_type,
                    .address_space = AsPrivatePhysical
                })
            });
        }
        case lea_op: {
            bool uniform = true;
            assert(prim_op.operands.count >= 2);
            const Node* base = prim_op.operands.nodes[0];
            uniform &= get_qualifier(base->type) == Uniform;
            const Type* curr_ptr_type = base->type;
            assert(without_qualifier(curr_ptr_type)->tag == PtrType_TAG && "lea expects a pointer as a base");

            const Node* offset = prim_op.operands.nodes[1];
            if (offset) {
                assert(without_qualifier(offset->type)->tag == Int_TAG && "lea expects an integer offset or NULL");
                const Type* pointee_type = without_qualifier(curr_ptr_type)->payload.ptr_type.pointed_type;

                assert(pointee_type->tag == ArrType_TAG && "if an offset is used, the base pointer must point to an array");
                uniform &= get_qualifier(offset->type) == Uniform;
            }

            // enter N levels of pointers
            size_t i = 2;
            while (true) {
                const Type* unqual_ptr_type = without_qualifier(curr_ptr_type);
                assert(unqual_ptr_type->tag == PtrType_TAG && "lea is supposed to work on, and yield pointers");
                if (i >= prim_op.operands.count) break;
                const Node* selector = prim_op.operands.nodes[i];
                assert(without_qualifier(selector->type)->tag == Int_TAG && "selectors must be integers");
                const Type* pointee_type = unqual_ptr_type->payload.ptr_type.pointed_type;
                assert(get_qualifier(pointee_type) == Unknown);
                switch (pointee_type->tag) {
                    case ArrType_TAG: {
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                            .pointed_type = pointee_type->payload.arr_type.element_type,
                            .address_space = unqual_ptr_type->payload.ptr_type.address_space
                        });
                        i++;
                        continue;
                    }
                    case RecordType_TAG: error("TODO"); // also remember to assert literals for the selectors !
                    default: error("lea selectors can only work on pointers to arrays or records")
                }
            }

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = uniform,
                .type = without_qualifier(curr_ptr_type)
            });
        }
        case reinterpret_op: {
            assert(prim_op.operands.count == 2);
            const Node* source = prim_op.operands.nodes[1];
            DivergenceQualifier qual;
            const Type* source_type = strip_qualifier(source->type, &qual);
            assert(qual != Unknown);
            const Type* target_type = prim_op.operands.nodes[0];

            assert(is_reinterpret_cast_legal(source_type, target_type));

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = qual == Uniform,
                .type = target_type
            });
        }
        case select_op: {
            assert(prim_op.operands.count == 3);
            assert(is_subtype(bool_type(arena), without_qualifier(prim_op.operands.nodes[0]->type)));
            // todo find true supertype
            assert(is_subtype(without_qualifier(prim_op.operands.nodes[1]->type), without_qualifier(prim_op.operands.nodes[2]->type)));

            return qualified_type(arena, (QualifiedType) {
                .is_uniform = (get_qualifier(prim_op.operands.nodes[1]->type) == Uniform) & (get_qualifier(prim_op.operands.nodes[2]->type) == Uniform),
                .type = without_qualifier(prim_op.operands.nodes[2]->type)
            });
        }
        case extract_dynamic_op:
        case extract_op: {
            assert(prim_op.operands.count >= 2);
            const Type* source = prim_op.operands.nodes[0];
            const Type* current_type = source->type;
            bool is_uniform = true;
            for (size_t i = 1; i < prim_op.operands.count; i++) {
                // Check index is valid !
                const Node* ith_index = prim_op.operands.nodes[i];
                bool dynamic_index = prim_op.op == extract_dynamic_op;
                if (dynamic_index) {
                    const Type* index_type = without_qualifier(ith_index->type);
                    assert(index_type->tag == Int_TAG && "extract_dynamic uses integers");
                } else {
                    assert(ith_index->tag == IntLiteral_TAG && "extract takes integer literals");
                }
                // Go down one level...
                is_uniform &= get_qualifier(current_type) != Varying;
                current_type = without_qualifier(current_type);
                switch(current_type->tag) {
                    case RecordType_TAG: {
                        assert(!dynamic_index);
                        size_t index_value = ith_index->payload.int_literal.value_i32;
                        assert(index_value < current_type->payload.record_type.members.count);
                        current_type = current_type->payload.record_type.members.nodes[index_value];
                        continue;
                    }
                    case ArrType_TAG: {
                        assert(!dynamic_index);
                        current_type = current_type->payload.arr_type.element_type;
                        continue;
                    }
                    case PackType_TAG: {
                        current_type = current_type->payload.pack_type.element_type;
                        continue;
                    }
                    default: error("Not a valid type to extract from")
                }
            }
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = is_uniform,
                .type = current_type
            });
        }
        case convert_op: {
            assert(prim_op.operands.count == 2);
            const Type* dst_type = prim_op.operands.nodes[0];
            bool is_uniform = without_qualifier(prim_op.operands.nodes[1]);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = is_uniform,
                .type = dst_type
            });
        }
        case empty_mask_op:
        case subgroup_active_mask_op: {
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = get_actual_mask_type(arena)
            });
        }
        case subgroup_ballot_op: {
            assert(prim_op.operands.count == 1);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = get_actual_mask_type(arena)
            });
        }
        case subgroup_elect_first_op: {
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = bool_type(arena)
            });
        }
        case subgroup_local_id_op: {
            assert(prim_op.operands.count == 0);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = false,
                .type = int32_type(arena)
            });
        }
        case subgroup_broadcast_first_op: {
            assert(prim_op.operands.count == 1);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = without_qualifier(prim_op.operands.nodes[0]->type)
            });
        }
        case mask_is_thread_active_op: {
            // TODO assert input is uniform
            assert(prim_op.operands.count == 2);
            return qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = bool_type(arena)
            });
        }
        default: error("unhandled primop %s", primop_names[prim_op.op]);
    }
}

static void check_uniform_helper(const Node* node, String errmsg) {
    assert(get_qualifier(node->type) == Uniform && errmsg);
}

#define HAS_PAYLOAD0(StructName, short_name)
#define HAS_PAYLOAD1(StructName, short_name) SHADY_UNUSED static const StructName* extract_##short_name(const Type* type) { \
    type = without_qualifier(type); \
    /*assert(type->tag == StructName##_TAG);*/ \
    if (type->tag != StructName##_TAG) return NULL; \
    return &type->payload.short_name; \
}

#define NODEDEF(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) HAS_PAYLOAD##has_payload(StructName, short_name)

TYPE_NODES()

static void check_arguments_types_against_parameters_helper(Nodes param_types, Nodes arg_types) {
    if (param_types.count != arg_types.count)
        error("Mismatched number of arguments/parameters");
    for (size_t i = 0; i < param_types.count; i++)
        check_subtype(param_types.nodes[i], arg_types.nodes[i]);
}

static Nodes check_callsite_helper(const Type* callee_type, Nodes argument_types) {
    const FnType* fn_type = extract_fn_type(callee_type);
    assert(fn_type);
    check_arguments_types_against_parameters_helper(fn_type->param_types, argument_types);

    return fn_type->return_types;
}

const Type* check_type_call_instr(IrArena* arena, Call call) {
    for (size_t i = 0; i < call.args.count; i++) {
        const Node* argument = call.args.nodes[i];
        assert(is_value(argument));
    }

    assert(get_qualifier(call.callee->type) == Uniform);
    return wrap_multiple_yield_types(arena, check_callsite_helper(call.callee->type, extract_types(arena, call.args)));
}

const Type* check_type_let(IrArena* arena, Let let) {
    //Nodes output_types = typecheck_instruction(arena, let.instruction);
    const Type* result_type = let.instruction->type;

    // check outputs
    Nodes var_tys = extract_variable_types(arena, &let.variables);
    switch (result_type->tag) {
        case Unit_TAG: error("You can only let-bind non-unit nodes");
        case RecordType_TAG: {
            if (result_type->payload.record_type.members.count != var_tys.count)
                error("let variables count != yield count from operation")
            for (size_t i = 0; i < var_tys.count; i++)
                check_subtype(var_tys.nodes[i], result_type->payload.record_type.members.nodes[i]);
            break;
        }
        default: {
            assert(var_tys.count == 1);
            check_subtype(var_tys.nodes[0], result_type);
            break;
        }
    }

    return unit_type(arena);
}

static void check_known_target_helper(const Node* target) {
    assert(target->tag == Function_TAG);
}

const Type* check_type_branch(IrArena* arena, Branch branch) {
    for (size_t i = 0; i < branch.args.count; i++) {
        const Node* argument = branch.args.nodes[i];
        assert(is_value(argument));
    }

    switch (branch.branch_mode) {
        case BrTailcall: {
            const PtrType* callee_type = extract_ptr_type(branch.target->type);
            assert(callee_type && "tail calls must have ptr callees");
            check_callsite_helper(callee_type->pointed_type, extract_types(arena, branch.args));
            return NULL;
        }
        case BrJump: {
            check_uniform_helper(branch.target, "Non-uniform branch targets are not allowed");
            check_callsite_helper(branch.target->type, extract_types(arena, branch.args));
            return NULL;
        }
        case BrIfElse: {
            assert(is_subtype(bool_type(arena), without_qualifier(branch.branch_condition->type)));

            const Node* branches[2] = { branch.true_target, branch.false_target };
            for (size_t i = 0; i < 2; i++) {
                check_uniform_helper(branches[i], "Non-uniform branch targets are not allowed");
                check_known_target_helper(branches[i]);
                check_callsite_helper(branches[i]->type, extract_types(arena, branch.args));
            }
            return NULL;
        }
        case BrSwitch: error("TODO")
    }

    // TODO check arguments and that both branches match
    return NULL;
}

const Type* check_type_join(IrArena* arena, Join join) {
    for (size_t i = 0; i < join.args.count; i++) {
        const Node* argument = join.args.nodes[i];
        assert(is_value(argument));
    }

    const Type* join_target_type;
    if (join.is_indirect) {
        const PtrType* ptr_type = extract_ptr_type(join.join_at->type);
        assert(ptr_type);
        join_target_type = ptr_type->pointed_type;
    } else {
        check_known_target_helper(join.join_at);
        join_target_type = join.join_at->type;
    }

    check_callsite_helper(join_target_type, extract_types(arena, join.args));

    return NULL;
}

const Type* check_type_callc(IrArena* arena, Callc callc) {
    for (size_t i = 0; i < callc.args.count; i++) {
        const Node* argument = callc.args.nodes[i];
        assert(is_value(argument));
    }

    const PtrType* callee_ptr_type = extract_ptr_type(callc.callee->type);
    assert(callee_ptr_type);
    const Nodes returned_types = check_callsite_helper(callee_ptr_type->pointed_type, extract_types(arena, callc.args));

    const Type* ret_cont_type;
    if (callc.is_return_indirect) {
        const PtrType* ret_cont_ptr_type = extract_ptr_type(callc.ret_cont->type);
        assert(ret_cont_ptr_type);
        ret_cont_type = ret_cont_ptr_type->pointed_type;
    } else {
        check_known_target_helper(callc.ret_cont);
        ret_cont_type = callc.ret_cont->type;
    }

    check_callsite_helper(ret_cont_type, returned_types);

    return NULL;
}

const Type* check_type_fn_addr(IrArena* arena, FnAddr fn_addr) {
    assert(fn_addr.fn->tag == Function_TAG);
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = ptr_type(arena, (PtrType) {
            .pointed_type = without_qualifier(fn_addr.fn->type),
            .address_space = AsProgramCode,
        })
    });
}

#pragma GCC diagnostic pop
