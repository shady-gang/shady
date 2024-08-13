#include "emit_c.h"

#include "../shady/type.h"
#include "../shady/analysis/cfg.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

static void emit_terminator(Emitter* emitter, FnEmitter* fn, Printer* block_printer, const Node* terminator);

String c_emit_body(Emitter* emitter, FnEmitter* fn, const Node* abs) {
    assert(abs && is_abstraction(abs));
    const Node* body = get_abstraction_body(abs);
    assert(body && is_terminator(body));
    CFNode* cf_node = cfg_lookup(fn->cfg, abs);
    Printer* p = open_growy_as_printer(new_growy());
    fn->instruction_printers[cf_node->rpo_index] = p;
    //indent(p);

    emit_terminator(emitter, fn, p, body);

    /*if (bbs && bbs->count > 0) {
        assert(emitter->config.dialect != CDialect_GLSL);
        error("TODO");
    }*/

    //deindent(p);
    // print(p, "\n");

    fn->instruction_printers[cf_node->rpo_index] = NULL;
    return printer_growy_unwrap(p);
}

static Strings emit_variable_declarations(Emitter* emitter, FnEmitter* fn, Printer* p, String given_name, Strings* given_names, Nodes types, bool mut, const Nodes* init_values) {
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
            CTerm initializer = c_emit_value(emitter, fn, init_values->nodes[i]);
            c_emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, &initializer);
        } else
            c_emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, NULL);
    }
    return strings(emitter->arena, types.count, names);
}

static void emit_if(Emitter* emitter, FnEmitter* fn, Printer* p, If if_) {
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, fn, p, "if_phi", NULL, if_.yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    assert(get_abstraction_params(if_.if_true).count == 0);
    String true_body = c_emit_body(&sub_emiter, fn, if_.if_true);
    CValue condition = to_cvalue(emitter, c_emit_value(emitter, fn, if_.condition));
    print(p, "\nif (%s) { ", condition);
    indent(p);
    print(p, "%s", true_body);
    deindent(p);
    print(p, "\n}");
    free_tmp_str(true_body);
    if (if_.if_false) {
        assert(get_abstraction_params(if_.if_false).count == 0);
        String false_body = c_emit_body(&sub_emiter, fn, if_.if_false);
        print(p, " else {");
        indent(p);
        print(p, "%s", false_body);
        deindent(p);
        print(p, "\n}");
        free_tmp_str(false_body);
    }

    Nodes results = get_abstraction_params(if_.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    print(p, "%s", c_emit_body(emitter, fn, if_.tail));
}

static void emit_match(Emitter* emitter, FnEmitter* fn, Printer* p, Match match) {
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, fn, p, "match_phi", NULL, match.yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    // Of course, the sensible thing to do here would be to emit a switch statement.
    // ...
    // Except that doesn't work, because C/GLSL have a baffling design wart: the `break` statement is overloaded,
    // meaning that if you enter a switch statement, which should be orthogonal to loops, you can't actually break
    // out of the outer loop anymore. Brilliant. So we do this terrible if-chain instead.
    //
    // We could do GOTO for C, but at the cost of arguably even more noise in the output, and two different codepaths.
    // I don't think it's quite worth it, just like it's not worth doing some data-flow based solution either.

    CValue inspectee = to_cvalue(emitter, c_emit_value(emitter, fn, match.inspect));
    bool first = true;
    LARRAY(CValue, literals, match.cases.count);
    for (size_t i = 0; i < match.cases.count; i++) {
        literals[i] = to_cvalue(emitter, c_emit_value(emitter, fn, match.literals.nodes[i]));
    }
    for (size_t i = 0; i < match.cases.count; i++) {
        String case_body = c_emit_body(&sub_emiter, fn, match.cases.nodes[i]);
        print(p, "\n");
        if (!first)
            print(p, "else ");
        print(p, "if (%s == %s) { ", inspectee, literals[i]);
        indent(p);
        print(p, "%s", case_body);
        deindent(p);
        print(p, "\n}");
        free_tmp_str(case_body);
        first = false;
    }
    if (match.default_case) {
        String default_case_body = c_emit_body(&sub_emiter, fn, match.default_case);
        print(p, "\nelse { ");
        indent(p);
        print(p, "%s", default_case_body);
        deindent(p);
        print(p, "\n}");
        free_tmp_str(default_case_body);
    }

    Nodes results = get_abstraction_params(match.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    print(p, "%s", c_emit_body(emitter, fn, match.tail));
}

static void emit_loop(Emitter* emitter, FnEmitter* fn, Printer* p, Loop loop) {
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
    Strings eparams = emit_variable_declarations(emitter, fn, p, NULL, &param_names, get_param_types(emitter->arena, params), true, &loop.initial_args);
    for (size_t i = 0; i < params.count; i++)
        register_emitted(&sub_emiter, fn, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, fn, p, "loop_break_phi", NULL, loop.yield_types, true, NULL);
    sub_emiter.phis.loop_break = ephis;

    String body = c_emit_body(&sub_emiter, fn, loop.body);
    print(p, "\nwhile(true) { ");
    indent(p);
    print(p, "%s", body);
    deindent(p);
    print(p, "\n}");
    free_tmp_str(body);

    Nodes results = get_abstraction_params(loop.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    print(p, "%s", c_emit_body(emitter, fn, loop.tail));
}

static void emit_terminator(Emitter* emitter, FnEmitter* fn, Printer* block_printer, const Node* terminator) {
    c_emit_mem(emitter, fn, get_terminator_mem(terminator));
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Join_TAG: error("this must be lowered away!");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
        case TailCall_TAG: error("TODO");
        case If_TAG: return emit_if(emitter, fn, block_printer, terminator->payload.if_instr);
        case Match_TAG: return emit_match(emitter, fn, block_printer, terminator->payload.match_instr);
        case Loop_TAG: return emit_loop(emitter, fn, block_printer, terminator->payload.loop_instr);
        case Control_TAG:      error("TODO")
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.args;
            if (args.count == 0) {
                print(block_printer, "\nreturn;");
            } else if (args.count == 1) {
                print(block_printer, "\nreturn %s;", to_cvalue(emitter, c_emit_value(emitter, fn, args.nodes[0])));
            } else {
                String packed = unique_name(emitter->arena, "pack_return");
                LARRAY(CValue, values, args.count);
                for (size_t i = 0; i < args.count; i++)
                    values[i] = to_cvalue(emitter, c_emit_value(emitter, fn, args.nodes[i]));
                c_emit_pack_code(block_printer, strings(emitter->arena, args.count, values), packed);
                print(block_printer, "\nreturn %s;", packed);
            }
            break;
        }
        case MergeSelection_TAG: {
            Nodes args = terminator->payload.merge_selection.args;
            Phis phis = emitter->phis.selection;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, fn, args.nodes[i])));

            break;
        }
        case MergeContinue_TAG: {
            Nodes args = terminator->payload.merge_continue.args;
            Phis phis = emitter->phis.loop_continue;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, fn, args.nodes[i])));
            print(block_printer, "\ncontinue;");
            break;
        }
        case MergeBreak_TAG: {
            Nodes args = terminator->payload.merge_break.args;
            Phis phis = emitter->phis.loop_break;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(block_printer, "\n%s = %s;", phis.strings[i], to_cvalue(emitter, c_emit_value(emitter, fn, args.nodes[i])));
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
