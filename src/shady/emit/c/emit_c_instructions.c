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

static Strings emit_variable_declarations(Emitter* emitter, Printer* p, String given_name, Strings* given_names, Nodes types) {
    if (given_names)
        assert(given_names->count == types.count);
    LARRAY(String, names, types.count);
    for (size_t i = 0; i < types.count; i++) {
        VarId id = fresh_id(emitter->arena);
        String name = given_names ? given_names->strings[i] : given_name;
        assert(name);
        names[i] = format_string(emitter->arena, "%s_%d", name, id);
        print(p, "\n%s;", c_emit_type(emitter, types.nodes[i], names[i]));
    }
    return strings(emitter->arena, types.count, names);
}

static void emit_primop(Emitter* emitter, Printer* p, const Node* node, InstructionOutputs outputs) {
    assert(node->tag == PrimOp_TAG);
    const PrimOp* prim_op = &node->payload.prim_op;
    enum {
        Infix, Prefix
    } m = Infix;
    String operator_str = NULL;
    CValue final_expression = NULL;
    switch (prim_op->op) {
        case assign_op:
        case subscript_op:
        case unit_op:
            return;
        case quote_op: {
            assert(outputs.count == 1);
            outputs.results[0] = emit_value(emitter, prim_op->operands.nodes[0]);
            outputs.needs_binding[0] = false;
            break;
        }
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
        case alloca_op:break;
        case alloca_logical_op:break;
        case load_op: {
            CAddr dereferenced = deref_term(emitter, emit_value(emitter, first(prim_op->operands)));
            outputs.results[0] = term_from_cvalue(dereferenced);
            outputs.needs_binding[0] = true;
            return;
        }
        case store_op: {
            CAddr dereferenced = deref_term(emitter, emit_value(emitter, first(prim_op->operands)));
            CValue cvalue = to_cvalue(emitter, emit_value(emitter, prim_op->operands.nodes[1]));
            print(p, "\n%s = %s;", dereferenced, cvalue);
            return;
        } case lea_op: {
            CTerm acc = emit_value(emitter, prim_op->operands.nodes[0]);

            const Type* t = extract_operand_type(prim_op->operands.nodes[0]->type);
            assert(t->tag == PtrType_TAG);

            const IntLiteral* offset_static_value = resolve_to_literal(prim_op->operands.nodes[1]);
            if (!offset_static_value || offset_static_value->value.i64 != 0) {
                CTerm offset = emit_value(emitter, prim_op->operands.nodes[1]);
                // we sadly need to drop to the value level (aka explicit pointer arithmetic) to do this
                // this means such code is never going to be legal in GLSL
                // also the cast is to account for our arrays-in-structs hack
                acc = term_from_cvalue(format_string(emitter->arena, "(%s) &(%s.arr[%s])", emit_type(emitter, t, NULL), deref_term(emitter, acc), to_cvalue(emitter, offset)));
            }

            t = t->payload.ptr_type.pointed_type;
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                switch (is_type(t)) {
                    case ArrType_TAG: {
                        CTerm index = emit_value(emitter, prim_op->operands.nodes[i]);
                        acc = term_from_cvar(format_string(emitter->arena, "(%s.arr[%s])", deref_term(emitter, acc), to_cvalue(emitter, index)));
                        break;
                    }
                    case RecordType_TAG: error("TODO");
                    default: error("lea can't work on this");
                }
            }
            assert(outputs.count == 1);
            outputs.results[0] = acc;
            outputs.needs_binding[0] = false;
            return;
        }
        case make_op: {
            CType t = emit_type(emitter, node->type, NULL);
            CValue src = to_cvalue(emitter, emit_value(emitter, first(prim_op->operands)));

            String bind_to = unique_name(emitter->arena, "make_body");
            print(p, "\n%s = %s;", emit_type(emitter, first(prim_op->operands)->type, bind_to), src);

            const Type* inside_type = extract_operand_type(first(prim_op->operands)->type);
            switch (inside_type->tag) {
                case RecordType_TAG: {
                    Growy* g = new_growy();
                    Printer* p2 = open_growy_as_printer(g);

                    Nodes field_types = inside_type->payload.record_type.members;
                    Strings field_names = inside_type->payload.record_type.names;
                    for (size_t i = 0; i < field_types.count; i++) {
                        if (field_names.count == field_types.count)
                            print(p2, "%s.%s", bind_to, field_names.strings[i]);
                        else
                            print(p2, "%s._%d", bind_to, i);

                        if (i + 1 < field_types.count)
                            print(p2, ", ");
                    }

                    final_expression = emit_compound_value(emitter, t, growy_data(g));

                    growy_destroy(g);
                    destroy_printer(p2);
                    break;
                }
                default: error("")
            }
            break;
        }
        case select_op: {
            assert(prim_op->operands.count == 3);
            CValue condition = to_cvalue(emitter, emit_value(emitter, prim_op->operands.nodes[0]));
            CValue l = to_cvalue(emitter, emit_value(emitter, prim_op->operands.nodes[1]));
            CValue r = to_cvalue(emitter, emit_value(emitter, prim_op->operands.nodes[2]));
            final_expression = format_string(emitter->arena, "(%s) ? (%s) : (%s)", condition, l, r);
            break;
        }
        case convert_op:break;
        case reinterpret_op: {
            assert(outputs.count == 1);
            CTerm src = emit_value(emitter, first(prim_op->operands));
            const Type* src_type = extract_operand_type(first(prim_op->operands)->type);
            const Type* dst_type = first(prim_op->type_arguments);
            if (emitter->config.dialect == GLSL) {
                if (is_glsl_scalar_type(src_type) && is_glsl_scalar_type(dst_type)) {
                    CType t = emit_type(emitter, prim_op->type_arguments.nodes[0], NULL);
                    outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "%s(%s)", t, to_cvalue(emitter, src)));
                    outputs.needs_binding[0] = false;
                } else
                    assert(false);
            } else if (src_type->tag == PtrType_TAG && dst_type->tag == PtrType_TAG || true) {
                CType t = emit_type(emitter, prim_op->type_arguments.nodes[0], NULL);
                outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "((%s) %s)", t, to_cvalue(emitter, src)));
                outputs.needs_binding[0] = false;
            } else {
                assert(false);
            }
            return;
        }
        case extract_dynamic_op:
        case extract_op: {
            CValue acc = to_cvalue(emitter, emit_value(emitter, first(prim_op->operands)));

            const Type* t = extract_operand_type(first(prim_op->operands)->type);
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
                        acc = format_string(emitter->arena, "(%s[%s])", acc, to_cvalue(emitter, emit_value(emitter, index)));
                        break;
                    }
                    default:
                    case NotAType: error("Must be a type");
                }
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
        case subgroup_elect_first_op: {
            final_expression = "true /* subgroup_elect_first */";
            break;
        }
        case subgroup_broadcast_first_op: {
            final_expression = format_string(emitter->arena, "%s /* subgroup_broadcast_first */", to_cvalue(emitter, emit_value(emitter, first(prim_op->operands))));
            break;
        }
        case subgroup_active_mask_op: {
            CType result_type = emit_type(emitter, node->type, NULL);
            final_expression = emit_compound_value(emitter, result_type, "1, 0, 0, 0");
            break;
        }
        case subgroup_ballot_op: {
            CType result_type = emit_type(emitter, node->type, NULL);
            CValue value = to_cvalue(emitter, emit_value(emitter, first(prim_op->operands)));
            final_expression = emit_compound_value(emitter, result_type, format_string(emitter->arena, "%s, 0, 0, 0", value));
            break;
        }
        case subgroup_local_id_op: {
            final_expression = "0 /* subgroup_local_idy */";
            break;
        }
        case empty_mask_op: {
            CType result_type = emit_type(emitter, node->type, NULL);
            final_expression = emit_compound_value(emitter, result_type, "0, 0, 0, 0");
            break;
        }
        case mask_is_thread_active_op: {
            CValue value = to_cvalue(emitter, emit_value(emitter, first(prim_op->operands)));
            switch (emitter->config.dialect) {
                case C: final_expression = format_string(emitter->arena, "(%s[0] == 1)", value); break;
                case GLSL: final_expression = format_string(emitter->arena, "(%s.x == 1)", value); break;
            }
            break;
        }
        case debug_printf_op: {
            CValue str = to_cvalue(emitter, emit_value(emitter, first(prim_op->operands)));
            print(p, "\nprintf(%s);", str);
            return;
        }
        case PRIMOPS_COUNT: assert(false); break;
    }

    assert(outputs.count == 1);
    outputs.needs_binding[0] = true;
    if (operator_str == NULL) {
        if (final_expression)
            outputs.results[0] = term_from_cvalue(final_expression);
        else
            outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "/* todo: implement %s */", primop_names[prim_op->op]));
        return;
    }

    switch (m) {
        case Infix: {
            CTerm a = emit_value(emitter, prim_op->operands.nodes[0]);
            CTerm b = emit_value(emitter, prim_op->operands.nodes[1]);
            outputs.results[0] = term_from_cvalue(
                    format_string(emitter->arena, "%s %s %s", to_cvalue(emitter, a), operator_str, to_cvalue(emitter, b)));
            break;
        }
        case Prefix: {
            CTerm operand = emit_value(emitter, prim_op->operands.nodes[0]);
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
        print(paramsp, to_cvalue(emitter, emit_value(emitter, args.nodes[i])));
        if (i + 1 < args.count)
            print(paramsp, ", ");
    }

    CValue callee;
    if (call->tag == LeafCall_TAG) {
        emit_decl(emitter, call->payload.leaf_call.callee);
        callee = to_cvalue(emitter, *lookup_existing_term(emitter, call->payload.leaf_call.callee));
    } else
        callee = to_cvalue(emitter, emit_value(emitter, call->payload.indirect_call.callee));

    String params = printer_growy_unwrap(paramsp);

    Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, call->type);
    assert(yield_types.count == outputs.count);
    if (yield_types.count > 1) {
        String named = unique_name(emitter->arena, "result");
        print(p, "\n%s = %s(%s);", emit_type(emitter, call->type, named), callee, params);
        for (size_t i = 0; i < yield_types.count; i++) {
            outputs.results[i] = term_from_cvalue(format_string(emitter->arena, "%s->_%d", named, i));
            outputs.needs_binding[i] = false;
        }
    } else if (yield_types.count == 1) {
        outputs.results[0] = term_from_cvalue(format_string(emitter->arena, "%s(%s)", callee, params));
        outputs.needs_binding[0] = true;
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
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, if_->yield_types);
    sub_emiter.phis.selection = ephis;

    assert(get_anonymous_lambda_params(if_->if_true).count == 0);
    String true_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_true), NULL);
    CValue condition = to_cvalue(emitter, emit_value(emitter, if_->condition));
    print(p, "\nif (%s) %s", condition, true_body);
    free_tmp_str(true_body);
    if (if_->if_false) {
        assert(get_anonymous_lambda_params(if_->if_false).count == 0);
        String false_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_false), NULL);
        print(p, " else %s", false_body);
        free_tmp_str(false_body);
    }

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.needs_binding[i] = false;
    }
}

static void emit_match(Emitter* emitter, Printer* p, const Node* match_instr, InstructionOutputs outputs) {
    assert(match_instr->tag == Match_TAG);
    const Match* match = &match_instr->payload.match_instr;
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, match->yield_types);
    sub_emiter.phis.selection = ephis;

    // Of course, the sensible thing to do here would be to emit a switch statement.
    // ...
    // Except that doesn't work, because C/GLSL have a baffling design wart: the `break` statement is overloaded,
    // meaning that if you enter a switch statement, which should be orthogonal to loops, you can't actually break
    // out of the outer loop anymore. Brilliant. So we do this terrible if-chain instead.
    //
    // We could do GOTO for C, but at the cost of arguably even more noise in the output, and two different codepaths.
    // I don't think it's quite worth it, just like it's not worth doing some data-flow based solution either.

    CValue inspectee = to_cvalue(emitter, emit_value(emitter, match->inspect));
    bool first = true;
    for (size_t i = 0; i < match->cases.count; i++) {
        String case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->cases.nodes[i]), NULL);
        CValue literal = to_cvalue(emitter, emit_value(emitter, match->literals.nodes[i]));
        print(p, "\n");
        if (!first)
            print(p, "else ");
        print(p, "if (%s == %s) %s", inspectee, literal, case_body);
        free_tmp_str(case_body);
        first = false;
    }
    if (match->default_case) {
        String default_case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->default_case), NULL);
        print(p, "\nelse %s", default_case_body);
        free_tmp_str(default_case_body);
    }

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.needs_binding[i] = false;
    }
}

static void emit_loop(Emitter* emitter, Printer* p, const Node* loop_instr, InstructionOutputs outputs) {
    assert(loop_instr->tag == Loop_TAG);
    const Loop* loop = &loop_instr->payload.loop_instr;

    Emitter sub_emiter = *emitter;
    Nodes params = get_anonymous_lambda_params(loop->body);
    Strings param_names = extract_variable_names(emitter->arena, params);
    Strings eparams = emit_variable_declarations(emitter, p, NULL, &param_names, extract_variable_types(emitter->arena, params));
    for (size_t i = 0; i < params.count; i++)
        register_emitted(&sub_emiter, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, loop->yield_types);
    sub_emiter.phis.loop_break = ephis;

    String body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(loop->body), NULL);
    print(p, "\nwhile(true) %s", body);
    free_tmp_str(body);

    assert(outputs.count == ephis.count);
    for (size_t i = 0; i < outputs.count; i++) {
        outputs.results[i] = term_from_cvalue(ephis.strings[i]);
        outputs.needs_binding[i] = false;
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
        case Instruction_Control_TAG: error("TODO")
    }
}
