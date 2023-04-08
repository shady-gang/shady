#include "emit_c.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include "../../type.h"
#include "../../ir_private.h"

#include <assert.h>
#include <stdlib.h>

#pragma GCC diagnostic error "-Wswitch"

void emit_pack_code(Printer* p, Strings src, String dst) {
    for (size_t i = 0; i < src.count; i++) {
        print(p, "\n%s->_%d = %s", dst, src.strings[i], i);
    }
}

void emit_unpack_code(Printer* p, String src, Strings dst) {
    for (size_t i = 0; i < dst.count; i++) {
        print(p, "\n%s = %s->_%d", dst.strings[i], src, i);
    }
}

static Strings emit_variable_declarations(Emitter* emitter, Printer* p, String given_name, Strings* given_names, Nodes types, const Nodes* init_values) {
    if (given_names)
        assert(given_names->count == types.count);
    if (init_values)
        assert(init_values->count == types.count);
    LARRAY(String, names, types.count);
    for (size_t i = 0; i < types.count; i++) {
        VarId id = fresh_id(emitter->arena);
        String name = given_names ? given_names->strings[i] : given_name;
        assert(name);
        names[i] = format_string(emitter->arena, "%s_%d", name, id);
        if (init_values)
            print(p, "\n%s = %s;", c_emit_type(emitter, types.nodes[i], names[i]), to_cvalue(emitter, emit_value(emitter, p, init_values->nodes[i])));
        else
            print(p, "\n%s;", c_emit_type(emitter, types.nodes[i], names[i]));
    }
    return strings(emitter->arena, types.count, names);
}

static void emit_primop(Emitter* emitter, Printer* p, const Node* node, InstructionOutputs outputs) {
    assert(node->tag == PrimOp_TAG);
    IrArena* arena = emitter->arena;
    const PrimOp* prim_op = &node->payload.prim_op;
    enum {
        Infix, Prefix
    } m = Infix;
    String operator_str = NULL;
    CValue final_expression = NULL;
    switch (prim_op->op) {
        case deref_op:
        case assign_op:
        case subscript_op: assert(false);
        case quote_op: {
            assert(outputs.count == 1);
            for (size_t i = 0; i < prim_op->operands.count; i++) {
                outputs.results[i] = emit_value(emitter, p, prim_op->operands.nodes[i]);
                outputs.binding[i] = NoBinding;
            }
            break;
        }
        case add_carry_op:
        case sub_borrow_op:
        case mul_extended_op:
            error("TODO: implement extended arithm ops in C");
            break;
        case add_op: operator_str = "+";  break;
        case sub_op: operator_str = "-";  break;
        case mul_op: operator_str = "*";  break;
        case div_op: operator_str = "/";  break;
        case mod_op: operator_str = "%";  break;
        case neg_op: operator_str = "-"; m = Prefix; break;
        case gt_op: operator_str = ">";  break;
        case gte_op: operator_str = ">="; break;
        case lt_op: operator_str = "<";  break;
        case lte_op: operator_str = "<="; break;
        case eq_op: operator_str = "=="; break;
        case neq_op: operator_str = "!="; break;
        case and_op: operator_str = "&";  break;
        case or_op: operator_str = "|";  break;
        case xor_op: operator_str = "^";  break;
        case not_op: operator_str = "!"; m = Prefix; break;
        // TODO achieve desired right shift semantics through unsigned/signed casts
        case rshift_logical_op:
            operator_str = ">>";
            break;
        case rshift_arithm_op:
            operator_str = ">>";
            break;
        case lshift_op:
            operator_str = "<<";
            break;
        // MATH OPS
        case sin_op:
        case cos_op:
        case abs_op:
        case sign_op:
        case floor_op:
        case ceil_op:
        case round_op:
        case fract_op:
        case sqrt_op:
        case inv_sqrt_op: error("TODO");
        case alloca_op:
        case alloca_subgroup_op: error("Lower me");
        case alloca_logical_op: {
            assert(outputs.count == 1);
            outputs.results[0] = (CTerm) { .value = NULL, .var = NULL };
            outputs.binding[0] = LetMutBinding;
            return;
        }
        case load_op: {
            CAddr dereferenced = deref_term(emitter, emit_value(emitter, p, first(prim_op->operands)));
            outputs.results[0] = term_from_cvalue(dereferenced);
            outputs.binding[0] = LetBinding;
            return;
        }
        case store_op: {
            CAddr dereferenced = deref_term(emitter, emit_value(emitter, p, first(prim_op->operands)));
            CValue cvalue = to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[1]));
            print(p, "\n%s = %s;", dereferenced, cvalue);
            return;
        } case lea_op: {
            CTerm acc = emit_value(emitter, p, prim_op->operands.nodes[0]);

            const Type* curr_ptr_type = get_unqualified_type(prim_op->operands.nodes[0]->type);
            assert(curr_ptr_type->tag == PtrType_TAG);

            const IntLiteral* offset_static_value = resolve_to_literal(prim_op->operands.nodes[1]);
            if (!offset_static_value || offset_static_value->value.i64 != 0) {
                CTerm offset = emit_value(emitter, p, prim_op->operands.nodes[1]);
                // we sadly need to drop to the value level (aka explicit pointer arithmetic) to do this
                // this means such code is never going to be legal in GLSL
                // also the cast is to account for our arrays-in-structs hack
                acc = term_from_cvalue(format_string(arena, "((%s) &(%s.arr[%s]))", emit_type(emitter, curr_ptr_type, NULL), deref_term(emitter, acc), to_cvalue(emitter, offset)));
            }

            //t = t->payload.ptr_type.pointed_type;
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                const Type* pointee_type = get_pointee_type(arena, curr_ptr_type);
                const Node* selector = prim_op->operands.nodes[i];
                switch (is_type(pointee_type)) {
                    case ArrType_TAG: {
                        CTerm index = emit_value(emitter, p, selector);
                        acc = term_from_cvar(format_string(arena, "(%s.arr[%s])", deref_term(emitter, acc), to_cvalue(emitter, index)));
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                                .pointed_type = pointee_type->payload.arr_type.element_type,
                                .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        break;
                    }
                    case TypeDeclRef_TAG: {
                        pointee_type = get_nominal_type_body(pointee_type);
                        SHADY_FALLTHROUGH
                    }
                    case RecordType_TAG: {
                        assert(selector->tag == IntLiteral_TAG && "selectors when indexing into a record need to be constant");
                        size_t static_index = get_int_literal_value(selector, false);
                        assert(static_index < pointee_type->payload.record_type.members.count);
                        Strings names = pointee_type->payload.record_type.names;
                        if (names.count == 0)
                            acc = term_from_cvar(format_string(arena, "(%s._%d)", deref_term(emitter, acc), static_index));
                        else
                            acc = term_from_cvar(format_string(arena, "(%s.%s)", deref_term(emitter, acc), names.strings[static_index]));
                        curr_ptr_type = ptr_type(arena, (PtrType) {
                                .pointed_type = pointee_type->payload.record_type.members.nodes[static_index],
                                .address_space = curr_ptr_type->payload.ptr_type.address_space
                        });
                        break;
                    }
                    default: error("lea can't work on this");
                }
            }
            assert(outputs.count == 1);
            outputs.results[0] = acc;
            outputs.binding[0] = emitter->config.dialect == ISPC ? LetBinding : NoBinding;
            outputs.binding[0] = NoBinding;
            return;
        }
        case memcpy_op: {
            print(p, "\nmemcpy(%s, %s, %s);", to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[0])), to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[1])), to_cvalue(emitter, c_emit_value(emitter, p, prim_op->operands.nodes[2])));
            return;
        }
        case size_of_op:
            final_expression = format_string(emitter->arena, "sizeof(%s)", c_emit_type(emitter, first(prim_op->type_arguments), NULL));
            break;
        case align_of_op:
            final_expression = format_string(emitter->arena, "alignof(%s)", c_emit_type(emitter, first(prim_op->type_arguments), NULL));
            break;
        case offset_of_op: {
            // TODO get member name
            String member_name; error("TODO");
            final_expression = format_string(emitter->arena, "offsetof(%s, %s)", c_emit_type(emitter, first(prim_op->type_arguments), NULL), member_name);
            break;
        } case select_op: {
            assert(prim_op->operands.count == 3);
            CValue condition = to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[0]));
            CValue l = to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[1]));
            CValue r = to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[2]));
            final_expression = format_string(emitter->arena, "(%s) ? (%s) : (%s)", condition, l, r);
            break;
        }
        case convert_op: error("TODO");
        case reinterpret_op: {
            assert(outputs.count == 1);
            CTerm src = emit_value(emitter, p, first(prim_op->operands));
            const Type* src_type = get_unqualified_type(first(prim_op->operands)->type);
            const Type* dst_type = first(prim_op->type_arguments);
            if (emitter->config.dialect == GLSL) {
                if (is_glsl_scalar_type(src_type) && is_glsl_scalar_type(dst_type)) {
                    CType t = emit_type(emitter, prim_op->type_arguments.nodes[0], NULL);
                    outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "%s(%s)", t, to_cvalue(emitter, src)));
                    outputs.binding[0] = NoBinding;
                } else
                    assert(false);
            } else if (src_type->tag == PtrType_TAG && dst_type->tag == PtrType_TAG) {
                CType t = emit_type(emitter, prim_op->type_arguments.nodes[0], NULL);
                outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "((%s) %s)", t, to_cvalue(emitter, src)));
                outputs.binding[0] = NoBinding;
            } else {
                assert(false);
            }
            return;
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            CValue acc = to_cvalue(emitter, emit_value(emitter, p, first(prim_op->operands)));
            bool insert = prim_op->op == insert_op;

            if (insert) {
                final_expression = unique_name(arena, "modified");
                print(p, "\n%s = %s;", c_emit_type(emitter, first(prim_op->operands)->type, final_expression), acc);
            }

            const Type* t = get_unqualified_type(first(prim_op->operands)->type);
            for (size_t i = 1; i < prim_op->operands.count; i++) {
                const Node* index = prim_op->operands.nodes[i];
                const IntLiteral* static_index = resolve_to_literal(index);

                switch (is_type(t)) {
                    case Type_TypeDeclRef_TAG: {
                        const Node* decl = t->payload.type_decl_ref.decl;
                        assert(decl && decl->tag == NominalType_TAG);
                        t = decl->payload.nom_type.body;
                        SHADY_FALLTHROUGH
                    }
                    case Type_RecordType_TAG: {
                        assert(static_index);
                        Strings names = t->payload.record_type.names;
                        if (names.count == 0)
                            acc = format_string(emitter->arena, "(%s._%d)", acc, static_index->value.u64);
                        else
                            acc = format_string(emitter->arena, "(%s.%s)", acc, names.strings[static_index->value.u64]);
                        break;
                    }
                    case Type_ArrType_TAG:
                    case Type_PackType_TAG: {
                        acc = format_string(emitter->arena, "(%s[%s])", acc, to_cvalue(emitter, emit_value(emitter, p, index)));
                        break;
                    }
                    default:
                    case NotAType: error("Must be a type");
                }
            }

            if (insert) {
                print(p, "\n%s = %s;", acc, to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[1])));
                break;
            }

            final_expression = acc;
            break;
        }
        case get_stack_base_op:
        case get_stack_base_uniform_op:
        case push_stack_op:
        case pop_stack_op:
        case push_stack_uniform_op:
        case pop_stack_uniform_op:
        case get_stack_pointer_op:
        case get_stack_pointer_uniform_op:
        case set_stack_pointer_op:
        case set_stack_pointer_uniform_op: error("Stack operations need to be lowered.");
        case default_join_point_op:
        case create_joint_point_op: error("lowered in lower_tailcalls.c");
        case subgroup_reduce_sum_op: {
            CValue value = to_cvalue(emitter, emit_value(emitter, p, first(prim_op->operands)));
            switch (emitter->config.dialect) {
                case ISPC: final_expression = format_string(emitter->arena, "reduce_add(%s)", value); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_elect_first_op: {
            switch (emitter->config.dialect) {
                case ISPC: final_expression = format_string(emitter->arena, "(programIndex == count_trailing_zeros(lanemask()))"); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_broadcast_first_op: {
            CValue value = to_cvalue(emitter, emit_value(emitter, p, first(prim_op->operands)));
            switch (emitter->config.dialect) {
                case ISPC: final_expression = format_string(emitter->arena, "extract(%s, count_trailing_zeros(lanemask()))", value); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_active_mask_op: {
            switch (emitter->config.dialect) {
                case ISPC: final_expression = "lanemask()"; break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_ballot_op: {
            CValue value = to_cvalue(emitter, emit_value(emitter, p, first(prim_op->operands)));
            switch (emitter->config.dialect) {
                case ISPC: final_expression = format_string(emitter->arena, "packmask(%s)", value); break;
                case C:
                case GLSL: error("TODO")
            }
            break;
        }
        case subgroup_id_op:
        case subgroup_local_id_op: {
            switch (emitter->config.dialect) {
                case ISPC: final_expression = "programIndex"; break;
                case C: error("TODO");
                case GLSL: error("TODO");
            }
            break;
        }
        case workgroup_id_op:
        case workgroup_local_id_op:
        case workgroup_num_op:
        case workgroup_size_op:
        case global_id_op: {
            error("TODO")
        }
        case empty_mask_op:
        case mask_is_thread_active_op: error("lower_me");
        case debug_printf_op: {
            String args_list = "";
            for (size_t i = 0; i < prim_op->operands.count; i++) {
                CValue str = to_cvalue(emitter, emit_value(emitter, p, prim_op->operands.nodes[i]));

                if (emitter->config.dialect == ISPC && i > 0)
                    str = format_string(emitter->arena, "extract(%s, printf_thread_index)", str);

                if (i > 0)
                    args_list = format_string(emitter->arena, "%s, %s", args_list, str);
                else
                    args_list = str;
            }
            switch (emitter->config.dialect) {
                case ISPC:
                    print(p, "\nforeach_active(printf_thread_index) { print(%s); }", args_list);
                    break;
                case C:
                    print(p, "\nprintf(%s);", args_list);
                    break;
                case GLSL: warn_print("printf is not supported in GLSL");
                    break;
            }

            return;
        }
        case PRIMOPS_COUNT: assert(false); break;
    }

    assert(outputs.count == 1);
    outputs.binding[0] = LetBinding;
    if (operator_str == NULL) {
        if (final_expression)
            outputs.results[0] = term_from_cvalue(final_expression);
        else
            outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "/* todo: implement %s */", primop_names[prim_op->op]));
        return;
    }

    switch (m) {
        case Infix: {
            CTerm a = emit_value(emitter, p, prim_op->operands.nodes[0]);
            CTerm b = emit_value(emitter, p, prim_op->operands.nodes[1]);
            outputs.results[0] = term_from_cvalue(
                    format_string(emitter->arena, "%s %s %s", to_cvalue(emitter, a), operator_str, to_cvalue(emitter, b)));
            break;
        }
        case Prefix: {
            CTerm operand = emit_value(emitter, p, prim_op->operands.nodes[0]);
            outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "%s%s", operator_str, to_cvalue(emitter, operand)));
            break;
        } default: assert(false);
    }
}

static void emit_call(Emitter* emitter, Printer* p, const Node* call, InstructionOutputs outputs) {
    Nodes args;
    if (call->tag == LeafCall_TAG)
        args = call->payload.leaf_call.args;
    else if (call->tag == IndirectCall_TAG)
        args = call->payload.indirect_call.args;
    else
        assert(false);

    Growy* g = new_growy();
    Printer* paramsp = open_growy_as_printer(g);
    for (size_t i = 0; i < args.count; i++) {
        print(paramsp, to_cvalue(emitter, emit_value(emitter, p, args.nodes[i])));
        if (i + 1 < args.count)
            print(paramsp, ", ");
    }

    CValue callee;
    if (call->tag == LeafCall_TAG) {
        emit_decl(emitter, call->payload.leaf_call.callee);
        callee = to_cvalue(emitter, *lookup_existing_term(emitter, call->payload.leaf_call.callee));
    } else
        callee = to_cvalue(emitter, emit_value(emitter, p, call->payload.indirect_call.callee));

    String params = printer_growy_unwrap(paramsp);

    Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, call->type);
    assert(yield_types.count == outputs.count);
    if (yield_types.count > 1) {
        String named = unique_name(emitter->arena, "result");
        print(p, "\n%s = %s(%s);", emit_type(emitter, call->type, named), callee, params);
        for (size_t i = 0; i < yield_types.count; i++) {
            outputs.results[i] = term_from_cvalue(format_string(emitter->arena, "%s->_%d", named, i));
            // we have let-bound the actual result already, and extracting their components can be done inline
            outputs.binding[i] = NoBinding;
        }
    } else if (yield_types.count == 1) {
        outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "%s(%s)", callee, params));
        outputs.binding[0] = LetBinding;
    } else {
        print(p, "\n%s(%s);", callee, params);
    }
    free_tmp_str(params);
}

static const Node* get_anonymous_lambda_body(const Node* lambda) {
    assert(is_anonymous_lambda(lambda));
    return lambda->payload.anon_lam.body;
}

static Nodes get_anonymous_lambda_params(const Node* lambda) {
    assert(is_anonymous_lambda(lambda));
    return lambda->payload.anon_lam.params;
}

static void emit_if(Emitter* emitter, Printer* p, const Node* if_instr, InstructionOutputs outputs) {
    assert(if_instr->tag == If_TAG);
    const If* if_ = &if_instr->payload.if_instr;
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "if_phi", NULL, if_->yield_types, NULL);
    sub_emiter.phis.selection = ephis;

    assert(get_anonymous_lambda_params(if_->if_true).count == 0);
    String true_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_true), NULL);
    CValue condition = to_cvalue(emitter, emit_value(emitter, p, if_->condition));
    print(p, "\nif (%s) { %s}", condition, true_body);
    free_tmp_str(true_body);
    if (if_->if_false) {
        assert(get_anonymous_lambda_params(if_->if_false).count == 0);
        String false_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_false), NULL);
        print(p, " else {%s}", false_body);
        free_tmp_str(false_body);
    }

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.binding[i] = NoBinding;
    }
}

static void emit_match(Emitter* emitter, Printer* p, const Node* match_instr, InstructionOutputs outputs) {
    assert(match_instr->tag == Match_TAG);
    const Match* match = &match_instr->payload.match_instr;
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "match_phi", NULL, match->yield_types, NULL);
    sub_emiter.phis.selection = ephis;

    // Of course, the sensible thing to do here would be to emit a switch statement.
    // ...
    // Except that doesn't work, because C/GLSL have a baffling design wart: the `break` statement is overloaded,
    // meaning that if you enter a switch statement, which should be orthogonal to loops, you can't actually break
    // out of the outer loop anymore. Brilliant. So we do this terrible if-chain instead.
    //
    // We could do GOTO for C, but at the cost of arguably even more noise in the output, and two different codepaths.
    // I don't think it's quite worth it, just like it's not worth doing some data-flow based solution either.

    CValue inspectee = to_cvalue(emitter, emit_value(emitter, p, match->inspect));
    bool first = true;
    LARRAY(CValue, literals, match->cases.count);
    for (size_t i = 0; i < match->cases.count; i++) {
        literals[i] = to_cvalue(emitter, emit_value(emitter, p, match->literals.nodes[i]));
    }
    for (size_t i = 0; i < match->cases.count; i++) {
        String case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->cases.nodes[i]), NULL);
        print(p, "\n");
        if (!first)
            print(p, "else ");
        print(p, "if (%s == %s) { %s}", inspectee, literals[i], case_body);
        free_tmp_str(case_body);
        first = false;
    }
    if (match->default_case) {
        String default_case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->default_case), NULL);
        print(p, "\nelse { %s}", default_case_body);
        free_tmp_str(default_case_body);
    }

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.binding[i] = NoBinding;
    }
}

static void emit_loop(Emitter* emitter, Printer* p, const Node* loop_instr, InstructionOutputs outputs) {
    assert(loop_instr->tag == Loop_TAG);
    const Loop* loop = &loop_instr->payload.loop_instr;

    Emitter sub_emiter = *emitter;
    Nodes params = get_anonymous_lambda_params(loop->body);
    Strings param_names = get_variable_names(emitter->arena, params);
    Strings eparams = emit_variable_declarations(emitter, p, NULL, &param_names, get_variables_types(emitter->arena, params), &loop_instr->payload.loop_instr.initial_args);
    for (size_t i = 0; i < params.count; i++)
        register_emitted(&sub_emiter, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, loop->yield_types, NULL);
    sub_emiter.phis.loop_break = ephis;

    String body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(loop->body), NULL);
    print(p, "\nwhile(true) { %s}", body);
    free_tmp_str(body);

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.binding[i] = NoBinding;
    }
}

void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction, InstructionOutputs outputs) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: assert(false);
        case Instruction_PrimOp_TAG:       emit_primop(emitter, p, instruction, outputs); break;
        case Instruction_LeafCall_TAG:
        case Instruction_IndirectCall_TAG: emit_call  (emitter, p, instruction, outputs); break;
        case Instruction_If_TAG:           emit_if    (emitter, p, instruction, outputs); break;
        case Instruction_Match_TAG:        emit_match (emitter, p, instruction, outputs); break;
        case Instruction_Loop_TAG:         emit_loop  (emitter, p, instruction, outputs); break;
        case Instruction_Control_TAG:      error("TODO")
        case Instruction_Block_TAG:        error("Should be eliminated by the compiler")
    }
}
