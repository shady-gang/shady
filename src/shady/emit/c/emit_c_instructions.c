#include "emit_c.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include "../../type.h"

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

Strings emit_variable_declarations(Emitter* emitter, Printer* p, Nodes vars) {
    LARRAY(String, names, vars.count);
    for (size_t i = 0; i < vars.count; i++) {
        const Variable* var = &vars.nodes[i]->payload.var;
        names[i] = format_string(emitter->arena, "%s_%d", var->name, var->id);
        print(p, "\n%s;", c_emit_type(emitter, var->type, names[i]));
    }
    return strings(emitter->arena, vars.count, names);
}

Strings emit_values(Emitter* emitter, Nodes values) {
    LARRAY(String, names, values.count);
    for (size_t i = 0; i < values.count; i++) {
        names[i] = emit_value(emitter, values.nodes[i]);
    }
    return strings(emitter->arena, values.count, names);
}

static void emit_primop(Emitter* emitter, Printer* p, const Node* node, Strings outputs) {
    assert(node->tag == PrimOp_TAG);
    const PrimOp* prim_op = &node->payload.prim_op;
    enum {
        Infix, Prefix
    } m = Infix;
    String s = NULL, rhs = NULL;
    switch (prim_op->op) {
        case unit_op:
            return;
        case quote_op: {
            rhs = emit_value(emitter, prim_op->operands.nodes[0]);
            break;
        }
        case add_op: s = "+";  break;
        case sub_op: s = "-";  break;
        case mul_op: s = "*";  break;
        case div_op: s = "/";  break;
        case mod_op: s = "%";  break;
        case neg_op: s = "-"; m = Prefix; break;
        case gt_op:  s = ">";  break;
        case gte_op: s = ">="; break;
        case lt_op:  s = "<";  break;
        case lte_op: s = "<="; break;
        case eq_op:  s = "=="; break;
        case neq_op: s = "!="; break;
        case and_op: s = "&";  break;
        case or_op:  s = "|";  break;
        case xor_op: s = "^";  break;
        case not_op: s = "!"; m = Prefix; break;
        case rshift_logical_op:break;
        case rshift_arithm_op:break;
        case lshift_op:break;
        case assign_op:break;
        case subscript_op:break;
        case alloca_op:break;
        case alloca_slot_op:break;
        case alloca_logical_op:break;
        case load_op: s = "*"; m = Prefix; break;
        case store_op:
            print(p, "\n*%s = %s;", emit_value(emitter, prim_op->operands.nodes[0]), emit_value(emitter, prim_op->operands.nodes[1]));
            return;
        case lea_op: {
            const char* acc = emit_value(emitter, prim_op->operands.nodes[0]);
            assert(acc);
            if (prim_op->operands.nodes[1])
                acc = format_string(emitter->arena, "&(%s[%s])", acc, emit_value(emitter, prim_op->operands.nodes[1]));
            const Type* t = extract_operand_type(prim_op->operands.nodes[0]->type);
            assert(t->tag == PtrType_TAG);
            t = t->payload.ptr_type.pointed_type;
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                switch (is_type(t)) {
                    case ArrType_TAG: {
                        acc = format_string(emitter->arena, "&(%s[%s])", acc, emit_value(emitter, prim_op->operands.nodes[i]));
                        break;
                    }
                    case RecordType_TAG: error("TODO");
                    default: error("lea can't work on this");
                }
            }
            rhs = acc;
            break;
        }
        case make_op: break;
        case select_op:break;
        case convert_op:break;
        case reinterpret_op: {
            rhs = format_string(emitter->arena, "(%s) %s", emit_type(emitter, prim_op->type_arguments.nodes[0], NULL), emit_value(emitter, prim_op->operands.nodes[0]));
            break;
        }
        case extract_op:break;
        case extract_dynamic_op:break;
        case push_stack_op:break;
        case pop_stack_op:break;
        case push_stack_uniform_op:break;
        case pop_stack_uniform_op:break;
        case get_stack_pointer_op:break;
        case get_stack_pointer_uniform_op:break;
        case set_stack_pointer_op:break;
        case set_stack_pointer_uniform_op:break;
        case subgroup_elect_first_op:break;
        case subgroup_broadcast_first_op:break;
        case subgroup_active_mask_op:break;
        case subgroup_ballot_op:break;
        case subgroup_local_id_op:break;
        case empty_mask_op:break;
        case mask_is_thread_active_op:break;
        case debug_printf_op:break;
        case PRIMOPS_COUNT: assert(false); break;
    }

    assert(outputs.count == 1);

    if (s == NULL) {
        if (rhs)
            print(p, "\n%s = %s;", outputs.strings[0], rhs);
        else
            print(p, "\n%s; /* todo: implement %s */", outputs.strings[0], primop_names[prim_op->op]);
        return;
    }

    switch (m) {
        case Infix: print(p, "\n%s = %s %s %s;", outputs.strings[0], emit_value(emitter, prim_op->operands.nodes[0]), s, emit_value(emitter, prim_op->operands.nodes[1])); break;
        case Prefix: print(p, "\n%s = %s%s;", outputs.strings[0], s, emit_value(emitter, prim_op->operands.nodes[0])); break;
    }
}

static void emit_call(Emitter* emitter, Printer* p, const Node* call_instr, Strings outputs) {
    assert(call_instr->tag == Call_TAG);
    const Call* call = &call_instr->payload.call_instr;
    Growy* g = new_growy();
    Printer* paramsp = open_growy_as_printer(g);
    for (size_t i = 0; i < call->args.count; i++) {
        print(paramsp, emit_value(emitter, call->args.nodes[i]));
        if (i + 1 < call->args.count)
            print(paramsp, ", ");
    }

    String callee = emit_value(emitter, call->callee);
    String params = printer_growy_unwrap(paramsp);

    Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, call_instr->type);
    assert(yield_types.count == outputs.count);
    if (yield_types.count > 1) {
        String named = unique_name(emitter->arena, "result");
        print(p, "\n%s = %s(%s);", emit_type(emitter, call_instr->type, named), callee, params);
        emit_unpack_code(p, named, outputs);
    } else if (yield_types.count == 1) {
        print(p, "\n%s = %s(%s);", emit_type(emitter, call_instr->type, outputs.strings[0]), callee, params);
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

static void emit_if(Emitter* emitter, Printer* p, const Node* if_instr, Strings outputs) {
    assert(if_instr->tag == If_TAG);
    const If* if_ = &if_instr->payload.if_instr;
    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    assert(get_anonymous_lambda_params(if_->if_true).count == 0);
    String true_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_true), NULL);
    print(p, "\nif (%s) %s", emit_value(emitter, if_->condition), true_body);
    free_tmp_str(true_body);
    if (if_->if_false) {
        assert(get_anonymous_lambda_params(if_->if_false).count == 0);
        String false_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(if_->if_false), NULL);
        print(p, " else %s", false_body);
        free_tmp_str(false_body);
    }
}

static void emit_match(Emitter* emitter, Printer* p, const Node* match_instr, Strings outputs) {
    assert(match_instr->tag == Match_TAG);
    const Match* match = &match_instr->payload.match_instr;
    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    print(p, "\nswitch (%s) {", emit_value(emitter, match->inspect));
    indent(p);
    for (size_t i = 0; i < match->cases.count; i++) {
        String case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->cases.nodes[i]), NULL);
        print(p, "\ncase %s: %s\n", emit_value(emitter, match->literals.nodes[i]), case_body);
        free_tmp_str(case_body);
    }
    if (match->default_case) {
        String default_case_body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(match->default_case), NULL);
        print(p, "\ndefault: %s\n", default_case_body);
        free_tmp_str(default_case_body);
    }
    deindent(p);
    print(p, "\n}");
}

static void emit_loop(Emitter* emitter, Printer* p, const Node* loop_instr, Strings outputs) {
    assert(loop_instr->tag == Loop_TAG);
    const Loop* loop = &loop_instr->payload.loop_instr;

    Nodes params = get_anonymous_lambda_params(loop->body);
    Strings eparams = emit_variable_declarations(emitter, p, params);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.loop_continue = eparams;
    sub_emiter.phis.loop_break = outputs;
    register_emitted_list(&sub_emiter, params, eparams);

    String body = emit_lambda_body(&sub_emiter, get_anonymous_lambda_body(loop->body), NULL);
    print(p, "\nwhile(true) %s", body);
    free_tmp_str(body);
}

void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction, Strings outputs) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: assert(false);
        case Instruction_PrimOp_TAG: emit_primop(emitter, p, instruction, outputs); break;
        case Instruction_Call_TAG:   emit_call  (emitter, p, instruction, outputs); break;
        case Instruction_If_TAG:     emit_if    (emitter, p, instruction, outputs); break;
        case Instruction_Match_TAG:  emit_match (emitter, p, instruction, outputs); break;
        case Instruction_Loop_TAG:   emit_loop  (emitter, p, instruction, outputs); break;
        case Instruction_Control_TAG: error("TODO")
    }
}
