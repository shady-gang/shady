#include "emit_c.h"

#include "../shady/analysis/cfg.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

static void emit_terminator(Emitter* emitter, FnEmitter* fn, Printer* block_printer, const Node* terminator);

String shd_c_emit_body(Emitter* emitter, FnEmitter* fn, const Node* abs) {
    assert(abs && is_abstraction(abs));
    const Node* body = get_abstraction_body(abs);
    assert(body && is_terminator(body));
    CFNode* cf_node = shd_cfg_lookup(fn->cfg, abs);
    Printer* p = shd_new_printer_from_growy(shd_new_growy());
    fn->instruction_printers[cf_node->rpo_index] = p;
    //indent(p);

    emit_terminator(emitter, fn, p, body);

    /*if (bbs && bbs->count > 0) {
        assert(emitter->config.dialect != CDialect_GLSL);
        error("TODO");
    }*/

    //deindent(p);
    // shd_print(p, "\n");

    fn->instruction_printers[cf_node->rpo_index] = NULL;
    String s2 = shd_printer_growy_unwrap(p);
    String s = shd_string(emitter->arena, s2);
    free((void*)s2);
    return s;
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
        names[i] = shd_make_unique_name(emitter->arena, name);
        if (init_values) {
            CTerm initializer = shd_c_emit_value(emitter, fn, init_values->nodes[i]);
            shd_c_emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, &initializer);
        } else
            shd_c_emit_variable_declaration(emitter, p, types.nodes[i], names[i], mut, NULL);
    }
    return shd_strings(emitter->arena, types.count, names);
}

static void emit_if(Emitter* emitter, FnEmitter* fn, Printer* p, If if_) {
    Emitter sub_emiter = *emitter;
    Strings ephis = emit_variable_declarations(emitter, fn, p, "if_phi", NULL, if_.yield_types, true, NULL);
    sub_emiter.phis.selection = ephis;

    assert(get_abstraction_params(if_.if_true).count == 0);
    String true_body = shd_c_emit_body(&sub_emiter, fn, if_.if_true);
    String false_body = if_.if_false ? shd_c_emit_body(&sub_emiter, fn, if_.if_false) : NULL;
    String tail = shd_c_emit_body(emitter, fn, if_.tail);
    CValue condition = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, if_.condition));
    shd_print(p, "\nif (%s) { ", condition);
    shd_printer_indent(p);
    shd_print(p, "%s", true_body);
    shd_printer_deindent(p);
    shd_print(p, "\n}");
    if (if_.if_false) {
        assert(get_abstraction_params(if_.if_false).count == 0);
        shd_print(p, " else {");
        shd_printer_indent(p);
        shd_print(p, "%s", false_body);
        shd_printer_deindent(p);
        shd_print(p, "\n}");
    }

    Nodes results = get_abstraction_params(if_.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        shd_c_register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    shd_print(p, "%s", tail);
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

    CValue inspectee = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, match.inspect));
    bool first = true;
    LARRAY(CValue, literals, match.cases.count);
    LARRAY(String, bodies, match.cases.count);
    String default_case_body = shd_c_emit_body(&sub_emiter, fn, match.default_case);
    String tail = shd_c_emit_body(emitter, fn, match.tail);
    for (size_t i = 0; i < match.cases.count; i++) {
        literals[i] = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, match.literals.nodes[i]));
        bodies[i] = shd_c_emit_body(&sub_emiter, fn, match.cases.nodes[i]);
    }
    for (size_t i = 0; i < match.cases.count; i++) {
        shd_print(p, "\n");
        if (!first)
            shd_print(p, "else ");
        shd_print(p, "if (%s == %s) { ", inspectee, literals[i]);
        shd_printer_indent(p);
        shd_print(p, "%s", bodies[i]);
        shd_printer_deindent(p);
        shd_print(p, "\n}");
        first = false;
    }
    if (match.default_case) {
        shd_print(p, "\nelse { ");
        shd_printer_indent(p);
        shd_print(p, "%s", default_case_body);
        shd_printer_deindent(p);
        shd_print(p, "\n}");
    }

    Nodes results = get_abstraction_params(match.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        shd_c_register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    shd_print(p, "%s", tail);
}

static void emit_loop(Emitter* emitter, FnEmitter* fn, Printer* p, Loop loop) {
    Emitter sub_emiter = *emitter;
    Nodes params = get_abstraction_params(loop.body);
    Nodes variables = params;
    LARRAY(String, arr, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        arr[i] = shd_get_node_name_unsafe(variables.nodes[i]);
        if (!arr[i])
            arr[i] = shd_make_unique_name(emitter->arena, "phi");
    }
    Strings param_names = shd_strings(emitter->arena, variables.count, arr);
    Strings eparams = emit_variable_declarations(emitter, fn, p, NULL, &param_names, shd_get_param_types(emitter->arena, params), true, &loop.initial_args);
    for (size_t i = 0; i < params.count; i++)
        shd_c_register_emitted(&sub_emiter, fn, params.nodes[i], term_from_cvalue(eparams.strings[i]));

    sub_emiter.phis.loop_continue = eparams;
    Strings ephis = emit_variable_declarations(emitter, fn, p, "loop_break_phi", NULL, loop.yield_types, true, NULL);
    sub_emiter.phis.loop_break = ephis;

    String body = shd_c_emit_body(&sub_emiter, fn, loop.body);
    String tail = shd_c_emit_body(emitter, fn, loop.tail);
    shd_print(p, "\nwhile(true) { ");
    shd_printer_indent(p);
    shd_print(p, "%s", body);
    shd_printer_deindent(p);
    shd_print(p, "\n}");

    Nodes results = get_abstraction_params(loop.tail);
    for (size_t i = 0; i < ephis.count; i++) {
        shd_c_register_emitted(emitter, fn, results.nodes[i], term_from_cvalue(ephis.strings[i]));
    }

    shd_print(p, "%s", tail);
}

static void emit_terminator(Emitter* emitter, FnEmitter* fn, Printer* block_printer, const Node* terminator) {
    shd_c_emit_mem(emitter, fn, get_terminator_mem(terminator));
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Join_TAG: shd_error("this must be lowered away!");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
        case IndirectTailCall_TAG: shd_error("TODO");
        case If_TAG: return emit_if(emitter, fn, block_printer, terminator->payload.if_instr);
        case Match_TAG: return emit_match(emitter, fn, block_printer, terminator->payload.match_instr);
        case Loop_TAG: return emit_loop(emitter, fn, block_printer, terminator->payload.loop_instr);
        case Control_TAG:      shd_error("TODO")
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.args;
            if (args.count == 0) {
                shd_print(block_printer, "\nreturn;");
            } else if (args.count == 1) {
                shd_print(block_printer, "\nreturn %s;", shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[0])));
            } else {
                String packed = shd_make_unique_name(emitter->arena, "pack_return");
                const Type* fn_type = fn->cfg->entry->node->type;
                assert(fn_type->tag == FnType_TAG);
                shd_c_emit_variable_declaration(emitter, block_printer, shd_maybe_multiple_return(emitter->arena, fn_type->payload.fn_type.return_types), packed, true, NULL);
                LARRAY(CValue, values, args.count);
                for (size_t i = 0; i < args.count; i++)
                    values[i] = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[i]));
                shd_c_emit_pack_code(block_printer, shd_strings(emitter->arena, args.count, values), packed);
                shd_print(block_printer, "\nreturn %s;", packed);
            }
            break;
        }
        case MergeSelection_TAG: {
            Nodes args = terminator->payload.merge_selection.args;
            Phis phis = emitter->phis.selection;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                shd_print(block_printer, "\n%s = %s;", phis.strings[i], shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[i])));

            break;
        }
        case MergeContinue_TAG: {
            Nodes args = terminator->payload.merge_continue.args;
            Phis phis = emitter->phis.loop_continue;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                shd_print(block_printer, "\n%s = %s;", phis.strings[i], shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[i])));
            shd_print(block_printer, "\ncontinue;");
            break;
        }
        case MergeBreak_TAG: {
            Nodes args = terminator->payload.merge_break.args;
            Phis phis = emitter->phis.loop_break;
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                shd_print(block_printer, "\n%s = %s;", phis.strings[i], shd_c_to_ssa(emitter, shd_c_emit_value(emitter, fn, args.nodes[i])));
            shd_print(block_printer, "\nbreak;");
            break;
        }
        case Terminator_Unreachable_TAG: {
            switch (emitter->backend_config.dialect) {
                case CDialect_CUDA:
                case CDialect_C11:
                    shd_print(block_printer, "\n__builtin_unreachable();");
                    break;
                case CDialect_ISPC:
                    shd_print(block_printer, "\nassert(false);");
                    break;
                case CDialect_GLSL:
                    shd_print(block_printer, "\n//unreachable");
                    break;
            }
            break;
        }
    }
}
