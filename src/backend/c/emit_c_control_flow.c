#include "emit_c.h"

#include "../shady/type.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

static Strings emit_variable_declarations(Emitter* emitter, Printer* p, String given_name, Strings* given_names, Nodes types, bool mut, const Nodes* init_values) {
    if (given_names)
        assert(given_names->count == types.count);
    if (init_values)
        assert(init_values->count == types.count);
    LARRAY(String, names, types.count);
    for (size_t i = 0; i < types.count; i++) {
        String name = given_names ? given_names->strings[i] : given_name;
        assert(name);
        names[i] = unique_name(emitter->arena, name);
        if (init_values) {
            CTerm initializer = emit_value(emitter, p, init_values->nodes[i]);
            emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, &initializer);
        } else
            emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, NULL);
    }
    return strings(emitter->arena, types.count, names);
}

static void emit_if(Emitter* emitter, Printer* p, If if_) {
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "if_phi", NULL, if_.yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    assert(get_abstraction_params(if_.if_true).count == 0);
    String true_body = emit_lambda_body(&sub_emiter, get_abstraction_body(if_.if_true), NULL);
    CValue condition = to_cvalue(emitter, emit_value(emitter, p, if_.condition));
    print(p, "\nif (%s) { %s}", condition, true_body);
    free_tmp_str(true_body);
    if (if_.if_false) {
        assert(get_abstraction_params(if_.if_false).count == 0);
        String false_body = emit_lambda_body(&sub_emiter, get_abstraction_body(if_.if_false), NULL);
        print(p, " else {%s}", false_body);
        free_tmp_str(false_body);
    }

    Nodes results = get_abstraction_params(if_.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    c_emit_terminator(emitter, p, get_abstraction_body(if_.tail));
}

static void emit_match(Emitter* emitter, Printer* p, Match match) {
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, p, "match_phi", NULL, match.yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    // Of course, the sensible thing to do here would be to emit a switch statement.
    // ...
    // Except that doesn't work, because C/GLSL have a baffling design wart: the `break` statement is overloaded,
    // meaning that if you enter a switch statement, which should be orthogonal to loops, you can't actually break
    // out of the outer loop anymore. Brilliant. So we do this terrible if-chain instead.
    //
    // We could do GOTO for C, but at the cost of arguably even more noise in the output, and two different codepaths.
    // I don't think it's quite worth it, just like it's not worth doing some data-flow based solution either.

    CValue inspectee = to_cvalue(emitter, emit_value(emitter, p, match.inspect));
    bool first = true;
    LARRAY(CValue, literals, match.cases.count);
    for (size_t i = 0; i < match.cases.count; i++) {
        literals[i] = to_cvalue(emitter, emit_value(emitter, p, match.literals.nodes[i]));
    }
    for (size_t i = 0; i < match.cases.count; i++) {
        String case_body = emit_lambda_body(&sub_emiter, get_abstraction_body(match.cases.nodes[i]), NULL);
        print(p, "\n");
        if (!first)
            print(p, "else ");
        print(p, "if (%s == %s) { %s}", inspectee, literals[i], case_body);
        free_tmp_str(case_body);
        first = false;
    }
    if (match.default_case) {
        String default_case_body = emit_lambda_body(&sub_emiter, get_abstraction_body(match.default_case), NULL);
        print(p, "\nelse { %s}", default_case_body);
        free_tmp_str(default_case_body);
    }

    Nodes results = get_abstraction_params(match.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    c_emit_terminator(emitter, p, get_abstraction_body(match.tail));
}

static void emit_loop(Emitter* emitter, Printer* p, Loop loop) {
    Emitter sub_emiter = *emitter;
    Nodes params = get_abstraction_params(loop.body);
    Nodes variables = params;
    LARRAY(String, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        arr[i] = get_value_name_unsafe(variables.nodes[i]);
        if (!arr[i])
            arr[i] = unique_name(emitter->arena, "phi");
    }
    Strings param_names = strings(emitter->arena, variables.count, arr);
    Strings eparams = emit_variable_declarations(emitter, p, NULL, &param_names, get_param_types(emitter->arena, params), true, &loop.initial_args);
    for (size_t i = 0; i < params.count; i++)
        register_emitted(&sub_emiter, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, p, "loop_break_phi", NULL, loop.yield_types, true, NULL);
    sub_emiter.phis.loop_break = ephis;

    String body = emit_lambda_body(&sub_emiter, get_abstraction_body(loop.body), NULL);
    print(p, "\nwhile(true) { %s}", body);
    free_tmp_str(body);

    Nodes results = get_abstraction_params(loop.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    c_emit_terminator(emitter, p, get_abstraction_body(loop.tail));
}

void c_emit_terminator(Emitter* emitter, Printer* block_printer, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Join_TAG: error("this must be lowered away!");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
        case TailCall_TAG: error("TODO");
        /*case Let_TAG: {
            const Node* instruction = get_let_instruction(terminator);

            // we declare N local variables in order to store the result of the instruction
            Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, instruction->type);

            LARRAY(CTerm, results, yield_types.count);
            LARRAY(InstrResultBinding, bindings, yield_types.count);
            InstructionOutputs ioutputs = {
                .count = yield_types.count,
                .results = results,
                .binding = bindings,
            };
            emit_instruction(emitter, block_printer, instruction, ioutputs);

            // Nodes vars = terminator->payload.let.variables;
            // assert(vars.count == yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                bool has_result = results[i].value || results[i].var;
                switch (bindings[i]) {
                    case NoBinding: {
                        assert(has_result && "unbound results can't be empty");
                        register_emitted(emitter, extract_multiple_ret_types_helper(instruction, i), results[i]);
                        break;
                    }
                    case LetBinding: {
                        String variable_name = get_value_name_unsafe(extract_multiple_ret_types_helper(instruction, i));

                        if (!variable_name)
                            variable_name = "";

                        String bind_to = unique_name(emitter->arena, variable_name);

                        const Type* t = yield_types.nodes[i];

                        if (has_result)
                            emit_variable_declaration(emitter, block_printer, t, bind_to, false, &results[i]);
                        else
                            emit_variable_declaration(emitter, block_printer, t, bind_to, false, NULL);

                        register_emitted(emitter, extract_multiple_ret_types_helper(instruction, i), term_from_cvalue(bind_to));
                        break;
                    }
                    default: assert(false);
                }
            }
            emit_terminator(emitter, block_printer, terminator->payload.let.in);

            break;
        }*/
        case If_TAG: return emit_if(emitter, block_printer, terminator->payload.if_instr);
        case Match_TAG: return emit_match(emitter, block_printer, terminator->payload.match_instr);
        case Loop_TAG: return emit_loop(emitter, block_printer, terminator->payload.loop_instr);
        case Control_TAG:      error("TODO")
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.args;
            if (args.count == 0) {
                print(block_printer, "\nreturn;");
            } else if (args.count == 1) {
                print(block_printer, "\nreturn %s;", to_cvalue(emitter, emit_value(emitter, block_printer, args.nodes[0])));
            } else {
                String packed = unique_name(emitter->arena, "pack_return");
                LARRAY(CValue, values, args.count);
                for (size_t i = 0; i < args.count; i++)
                    values[i] = to_cvalue(emitter, emit_value(emitter, block_printer, args.nodes[i]));
                emit_pack_code(block_printer, strings(emitter->arena, args.count, values), packed);
                print(block_printer, "\nreturn %s;", packed);
            }
            break;
        }
        case MergeSelection_TAG: {
            Nodes args = terminator->payload.merge_selection.args;
            Phis phis = emitter->phis.selection;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, emit_value(emitter, block_printer, args.nodes[i])));

            break;
        }
        case MergeContinue_TAG: {
            Nodes args = terminator->payload.merge_continue.args;
            Phis phis = emitter->phis.loop_continue;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, emit_value(emitter, block_printer, args.nodes[i])));
            print(block_printer, "\ncontinue;");
            break;
        }
        case MergeBreak_TAG: {
            Nodes args = terminator->payload.merge_break.args;
            Phis phis = emitter->phis.loop_break;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, emit_value(emitter, block_printer, args.nodes[i])));
            print(block_printer, "\nbreak;");
            break;
        }
        case Terminator_Unreachable_TAG: {
            switch (emitter->config.dialect) {
                case CDialect_CUDA:
                case CDialect_C11:
                    print(block_printer, "\n__builtin_unreachable();");
                    break;
                case CDialect_ISPC:
                    print(block_printer, "\nassert(false);");
                    break;
                case CDialect_GLSL:
                    print(block_printer, "\n//unreachable");
                    break;
            }
            break;
        }
    }
}
