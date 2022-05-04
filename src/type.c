#include "type.h"

#include "log.h"
#include "local_array.h"

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
            if (supertype->payload.fn.atttributes.is_continuation != type->payload.fn.atttributes.is_continuation)
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

const Type* strip_qualifier(const Type* type, DivergenceQualifier* qual_out) {
    if (type->tag == QualifiedType_TAG) {
        *qual_out = type->payload.qualified_type.is_uniform ? Uniform : Varying;
        return type->payload.qualified_type.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

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

const Type* derive_fn_type(IrArena* arena, const Function* fn) {
    return fn_type(arena, (FnType) { .is_continuation = fn->atttributes.is_continuation, .param_types = extract_variable_types(arena, &fn->params), .return_types = fn->return_types });
}

const Type* check_type_fn(IrArena* arena, Function fn) {
    assert(!fn.atttributes.is_continuation || fn.return_types.count == 0);
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = derive_fn_type(arena, &fn)
    });
}

static bool is_as_access_uniform(AddressSpace as) {
    switch (as) {
        case AsGeneric: return false;
        case AsPrivate: return false;
        case AsShared:   return true;
        case AsGlobal:   return true;
        case AsInput:   return false;
        case AsOutput:  return false;
        case AsExternal: return true;
    }
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

const Type* check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

const Type* check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = int_type(arena)
    });
}

const Type* check_type_true_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }
const Type* check_type_false_lit(IrArena* arena) { return qualified_type(arena, (QualifiedType) { .type = bool_type(arena), .is_uniform = true }); }

const Type* check_type_fn_ret(IrArena* arena, Return ret) {
    // TODO check it then !
    return NULL;
}

Nodes typecheck_if_instr(IrArena* arena, If if_instr) {
    if (without_qualifier(if_instr.condition->type) != bool_type(arena))
        error("condition of a selection should be bool");
    // TODO check the contained Merge instrs
    return if_instr.yield_types;
}

Nodes typecheck_loop_instr(IrArena* arena, Loop loop_instr) {
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return loop_instr.yield_types;
}

Nodes typecheck_match_instr(IrArena* arena, Match match_instr) {
    // TODO check param against initial_args
    // TODO check the contained Merge instrs
    return match_instr.yield_types;
}

/// Checks the operands to a Primop and returns the produced types
Nodes typecheck_primop(IrArena* arena, PrimOp prim_op) {
    switch (prim_op.op) {
        case add_op:
        case sub_op:
        case mul_op:
        case div_op:
        case mod_op:
        {
             bool is_result_uniform = true;
             for (size_t i = 0; i < prim_op.operands.count; i++) {
                 const Node* arg = prim_op.operands.nodes[i];
                 DivergenceQualifier op_div;
                 const Type* arg_actual_type = strip_qualifier(arg->type, &op_div);
                 assert(op_div != Unknown); // we expect all operands to be clearly known !
                 is_result_uniform ^= op_div == Uniform;
                 // we work with numerical operands
                 assert(arg_actual_type == int_type(arena) && "todo improve this check");
             }

            return singleton(qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = int_type(arena) }));
        }
        case lt_op:
        case lte_op:
        case gt_op:
        case gte_op:
        case eq_op:
        case neq_op:
        {
            bool is_result_uniform = true;
            for (size_t i = 0; i < prim_op.operands.count; i++) {
                const Node* arg = prim_op.operands.nodes[i];
                DivergenceQualifier op_div;
                // TODO ensure these guys are compatible ?
                const Type* arg_actual_type = strip_qualifier(arg->type, &op_div);
                assert(op_div != Unknown); // we expect all operands to be clearly known !
                is_result_uniform ^= op_div == Uniform;
            }
            return singleton(qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = bool_type(arena) }));
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
            return empty();
        }
        case pop_stack_op:
        case pop_stack_uniform_op: {
            assert(prim_op.operands.count == 1);
            const Type* element_type = prim_op.operands.nodes[0];
            assert(get_qualifier(element_type) == Unknown && "annotations do not go here");
            return singleton(qualified_type(arena, (QualifiedType) { .is_uniform = prim_op.op == pop_stack_uniform_op, .type = element_type}));
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
            return singleton(qualified_type(arena, (QualifiedType) {
                .type = elem_type,
                .is_uniform = qual == Uniform
            }));
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
            return empty();
        }
        case alloca_op: {
            assert(prim_op.operands.count == 1);
            const Type* elem_type = prim_op.operands.nodes[0];
            assert(is_type(elem_type));
            return singleton(qualified_type(arena, (QualifiedType) {
                .is_uniform = true,
                .type = ptr_type(arena, (PtrType) {
                    .pointed_type = elem_type,
                    .address_space = AsPrivate
                })
            }));
        }
        case lea_op: {
            bool uniform = true;
            assert(prim_op.operands.count >= 2);
            const Node* offset = prim_op.operands.nodes[1];
            uniform &= get_qualifier(offset) == Uniform;
            assert(without_qualifier(offset->type)->tag == Int_TAG && "offset must be integer");

            const Node* base = prim_op.operands.nodes[0];
            uniform &= get_qualifier(base) == Uniform;

            // enter N levels of pointers
            const Type* curr_ptr_type = base->type;
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
                            .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        continue;
                    }
                    case RecordType_TAG: error("TODO"); // also remember to assert literals for the selectors !
                    default: error("lea selectors can only work on pointers to arrays or records")
                }
            }

            return singleton(qualified_type(arena, (QualifiedType) {
                .is_uniform = uniform,
                .type = curr_ptr_type
            }));
        }
        default: error("unhandled primop %s", primop_names[prim_op.op]);
    }
}

static void check_uniform_helper(const Node* node, String errmsg) {
    assert(get_qualifier(node->type) == Uniform && errmsg);
}

static void check_known_target_helper(const Node* target) {
    assert(target->tag == Function_TAG);
}

static const FnType* check_is_fn_and_get_fn_type_helper(const Node* node) {
    const Type* callee_type = without_qualifier(node->type);
    assert(callee_type->tag == FnType_TAG);
    return &callee_type->payload.fn_type;
}

static void check_cont_fn_helper(const Node* target, bool should_be_continuation) {
    const FnType* tgt_type = &without_qualifier(target->type)->payload.fn_type;
    assert(tgt_type->is_continuation == should_be_continuation);
}

static void check_arguments_against_parameters_helper(Nodes param_types, Nodes arguments) {
    if (param_types.count != arguments.count)
        error("Mismatched number of arguments/parameters");
    for (size_t i = 0; i < param_types.count; i++)
        check_subtype(param_types.nodes[i], arguments.nodes[i]->type);
}

static void check_arguments_types_against_parameters_helper(Nodes param_types, Nodes arg_types) {
    if (param_types.count != arg_types.count)
        error("Mismatched number of arguments/parameters");
    for (size_t i = 0; i < param_types.count; i++)
        check_subtype(param_types.nodes[i], arg_types.nodes[i]);
}

static Nodes check_callsite_helper(const Node* callee, Nodes arguments) {
    const FnType* fn_type = check_is_fn_and_get_fn_type_helper(callee);
    check_arguments_against_parameters_helper(fn_type->param_types, arguments);

    return fn_type->return_types;
}

Nodes typecheck_call(IrArena* arena, Call call) {
    assert(get_qualifier(call.callee->type) == Uniform);
    return check_callsite_helper(call.callee, call.args);
}

Nodes typecheck_instruction(IrArena* arena, const Node* instr) {
    switch (instr->tag) {
        case PrimOp_TAG: return typecheck_primop(arena, instr->payload.prim_op);
        case Call_TAG:   return typecheck_call(arena, instr->payload.call_instr);
        case If_TAG:     return typecheck_if_instr(arena, instr->payload.if_instr);
        case Loop_TAG:   return typecheck_loop_instr(arena, instr->payload.loop_instr);
        case Match_TAG:  return typecheck_match_instr(arena, instr->payload.match_instr);
        default:         error("unhandled instruction");
    }
    SHADY_UNREACHABLE;
}

const Type* check_type_let(IrArena* arena, Let let) {
    Nodes output_types = typecheck_instruction(arena, let.instruction);

    // check outputs
    Nodes var_tys = extract_variable_types(arena, &let.variables);
    if (output_types.count != var_tys.count)
        error("let variables count != yield count from operation")
    for (size_t i = 0; i < var_tys.count; i++)
        check_subtype(var_tys.nodes[i], output_types.nodes[i]);
    return NULL;
}

const Type* check_type_jump(IrArena* arena, Jump jump) {
    check_uniform_helper(jump.target, "Non-uniform jump targets are not allowed");
    check_known_target_helper(jump.target);
    check_cont_fn_helper(jump.target, true);
    check_callsite_helper(jump.target, jump.args);
    return NULL;
}

const Type* check_type_branch(IrArena* arena, Branch branch) {
    const Node* branches[2] = { branch.true_target, branch.false_target };
    for (size_t i = 0; i < 2; i++) {
        check_uniform_helper(branches[i], "Non-uniform branch targets are not allowed");
        check_known_target_helper(branches[i]);
        check_cont_fn_helper(branches[i], true);
        check_callsite_helper(branches[i], branch.args);
    }

    // TODO check arguments and that both branches match
    return NULL;
}

const Type* check_type_callf(IrArena* arena, Callf callf) {
    const FnType* ret_fn_type = check_is_fn_and_get_fn_type_helper(callf.ret_fn);
    check_cont_fn_helper(callf.ret_fn, false);

    const Nodes returned_types = check_callsite_helper(callf.callee, callf.args);
    check_arguments_types_against_parameters_helper(ret_fn_type->param_types, returned_types);

    return NULL;
}

const Type* check_type_callc(IrArena* arena, Callc callc) {
    const FnType* ret_cont_type = check_is_fn_and_get_fn_type_helper(callc.ret_cont);
    check_cont_fn_helper(callc.ret_cont, true);
    check_known_target_helper(callc.ret_cont);

    const Nodes returned_types = check_callsite_helper(callc.callee, callc.args);
    check_arguments_types_against_parameters_helper(ret_cont_type->param_types, returned_types);

    return NULL;
}

#pragma GCC diagnostic pop
