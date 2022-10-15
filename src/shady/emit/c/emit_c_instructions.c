#include "emit_c.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include "../../type.h"

#include <assert.h>
#include <stdlib.h>

void emit_pack_code(Emitter* e, Printer* p, const Nodes* src, String dst) {
    for (size_t i = 0; i < src->count; i++) {
        print(p, "\n%s->_%d = %s", dst, emit_value(e, src->nodes[i]), i);
    }
}

void emit_unpack_code(Emitter* e, Printer* p, String src, const Nodes* dst) {
    for (size_t i = 0; i < dst->count; i++) {
        print(p, "\n%s = %s->_%d", emit_value(e, dst->nodes[i]), src, i);
    }
}

static void declare_variables_helper(Emitter* emitter, Printer* p, const Nodes* vars) {
    for (size_t i = 0; i < vars->count; i++) {
        const Variable* var = &vars->nodes[i]->payload.var;
        String named = format_string(emitter->arena, "%s_%d", var->name, var->id);
        print(p, "\n%s;", c_emit_type(emitter, var->type, named));
        insert_dict(const Node*, String, emitter->emitted, vars->nodes[i], named);
    }
}

static void emit_primop(Emitter* emitter, Printer* p, const PrimOp* prim_op, const Nodes* outputs) {
    enum {
        Infix, Prefix
    } m = Infix;
    String s = NULL, rhs = NULL;
    switch (prim_op->op) {
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
            break;
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
        case select_op:break;
        case convert_op:break;
        case reinterpret_op: {
            rhs = format_string(emitter->arena, "(%s) %s", emit_type(emitter, prim_op->operands.nodes[0], NULL), emit_value(emitter, prim_op->operands.nodes[1]));
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

    if (outputs == NULL || outputs->count == 0)
        return;

    LARRAY(const char*, named, outputs->count);
    for (size_t i = 0; i < outputs->count; i++) {
        const Variable* var = &outputs->nodes[i]->payload.var;
        named[i] = format_string(emitter->arena, "%s_%d", var->name, var->id);
        insert_dict(const Node*, String, emitter->emitted, outputs->nodes[i], named[i]);
    }

    assert(outputs->count == 1);
    const Variable* var = &outputs->nodes[0]->payload.var;
    String decl = emit_type(emitter, outputs->nodes[0]->type, format_string(emitter->arena, "const %s_%d", var->name, var->id));

    if (s == NULL) {
        if (rhs)
            print(p, "\n%s = %s;", decl, rhs);
        else
            print(p, "\n%s; /* todo: implement %s */", decl, primop_names[prim_op->op]);
        return;
    }

    switch (m) {
        case Infix: print(p, "\n%s = %s %s %s;", decl, emit_value(emitter, prim_op->operands.nodes[0]), s, emit_value(emitter, prim_op->operands.nodes[1])); break;
        case Prefix: print(p, "\n%s = %s%s;", decl, s, emit_value(emitter, prim_op->operands.nodes[0])); break;
    }
}

static String emit_callee(Emitter* e, const Node* callee) {
    if (callee->tag == Lambda_TAG)
        return callee->payload.lam.name;
    else
        return emit_value(e, callee);
}

static void emit_call(Emitter* emitter, Printer* p, const Call* call, const Type* result_type, const Nodes* outputs) {
    Growy* g = new_growy();
    Printer* paramsp = open_growy_as_printer(g);
    for (size_t i = 0; i < call->args.count; i++) {
        print(paramsp, emit_value(emitter, call->args.nodes[i]));
        if (i + 1 < call->args.count)
            print(paramsp, ", ");
    }
    String params = printer_growy_unwrap(paramsp);

    if (outputs->count > 1) {
        declare_variables_helper(emitter, p, outputs);
        String named = unique_name(emitter->arena, "result");
        print(p, "\n%s = %s(%s);", emit_type(emitter, result_type, named), emit_callee(emitter, call->callee), params);
        emit_unpack_code(emitter, p, named, outputs);
    } else if (outputs->count == 1) {
        const Variable* var = &outputs->nodes[0]->payload.var;
        String named = format_string(emitter->arena, "%s_%d", var->name, var->id);
        print(p, "\n%s = %s(%s);", emit_type(emitter, var->type, named), emit_callee(emitter, call->callee), params);
        insert_dict(const Node*, String, emitter->emitted, outputs->nodes[0], named);
    } else {
        print(p, "\n%s(%s);", emit_callee(emitter, call->callee), params);
    }
    free(params);
}

static void emit_if(Emitter* emitter, Printer* p, const If* if_instr, const Nodes* outputs) {
    if (outputs->count > 0)
        print(p, "\n/* if yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    String true_body = emit_body(&sub_emiter, if_instr->if_true, NULL);
    String false_body = if_instr->if_false ? emit_body(&sub_emiter, if_instr->if_false, NULL) : NULL;
    print(p, "\nif (%s) %s", emit_value(emitter, if_instr->condition), true_body);
    if (false_body)
        print(p, " else %s", false_body);
    free(true_body);
    free(false_body);
}

static void emit_match(Emitter* emitter, Printer* p, const Match* match_instr, const Nodes* outputs) {
    if (outputs->count > 0)
        print(p, "\n/* match yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    print(p, "\nswitch (%s) {", emit_value(emitter, match_instr->inspect));
    indent(p);
    for (size_t i = 0; i < match_instr->cases.count; i++) {
        String case_body = emit_body(&sub_emiter, match_instr->cases.nodes[i], NULL);
        print(p, "\ncase %s: %s\n", emit_value(emitter, match_instr->literals.nodes[i]), case_body);
        free(case_body);
    }
    if (match_instr->default_case) {
        String default_case_body = emit_body(&sub_emiter, match_instr->default_case, NULL);
        print(p, "\ndefault: %s\n", default_case_body);
        free(default_case_body);
    }
    deindent(p);
    print(p, "\n}");
}

static void emit_loop(Emitter* emitter, Printer* p, const Loop* loop_instr, const Nodes* outputs) {
    if (loop_instr->params.count > 0)
        print(p, "\n/* loop parameters */");
    declare_variables_helper(emitter, p, &loop_instr->params);
    if (outputs->count > 0)
        print(p, "\n/* loop yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.loop_continue = &loop_instr->params;
    sub_emiter.phis.loop_break = outputs;

    String body = emit_body(&sub_emiter, loop_instr->body, NULL);
    print(p, "\nwhile(true) %s", body);
    free(body);
}

void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction) {
    assert(is_instruction(instruction));
    Nodes vars = nodes(emitter->arena, 0, NULL);

    if (instruction->tag == Let_TAG) {
        vars = instruction->payload.let.variables;
        instruction = instruction->payload.let.instruction;
    }

    switch (is_instruction(instruction)) {
        case Instruction_Let_TAG:
        case NotAnInstruction: assert(false);
        case Instruction_PrimOp_TAG: emit_primop(emitter, p, &instruction->payload.prim_op, &vars);     break;
        case Instruction_Call_TAG:   emit_call  (emitter, p, &instruction->payload.call_instr, instruction->type, &vars);  break;
        case Instruction_If_TAG:     emit_if    (emitter, p, &instruction->payload.if_instr, &vars);    break;
        case Instruction_Match_TAG:  emit_match (emitter, p, &instruction->payload.match_instr, &vars); break;
        case Instruction_Loop_TAG:   emit_loop  (emitter, p, &instruction->payload.loop_instr, &vars);  break;
    }
}
