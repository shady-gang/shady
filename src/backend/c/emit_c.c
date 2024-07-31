#include "emit_c.h"

#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/compile.h"
#include "../shady/transform/ir_gen_helpers.h"

#include "shady_cuda_prelude_src.h"
#include "shady_cuda_builtins_src.h"
#include "shady_glsl_120_polyfills_src.h"

#include "portability.h"
#include "dict.h"
#include "log.h"
#include "util.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#pragma GCC diagnostic error "-Wswitch"

static void emit_terminator(Emitter* emitter, Printer* block_printer, const Node* terminator);

CValue to_cvalue(SHADY_UNUSED Emitter* e, CTerm term) {
    if (term.value)
        return term.value;
    if (term.var)
        return format_string_arena(e->arena->arena, "(&%s)", term.var);
    assert(false);
}

CAddr deref_term(Emitter* e, CTerm term) {
    if (term.value)
        return format_string_arena(e->arena->arena, "(*%s)", term.value);
    if (term.var)
        return term.var;
    assert(false);
}

// TODO: utf8
static bool is_legal_c_identifier_char(char c) {
    if (c >= '0' && c <= '9')
        return true;
    if (c >= 'a' && c <= 'z')
        return true;
    if (c >= 'A' && c <= 'Z')
        return true;
    if (c == '_')
        return true;
    return false;
}

String legalize_c_identifier(Emitter* e, String src) {
    if (!src)
        return "unnamed";
    size_t len = strlen(src);
    LARRAY(char, dst, len + 1);
    size_t i;
    for (i = 0; i < len; i++) {
        char c = src[i];
        if (is_legal_c_identifier_char(c))
            dst[i] = c;
        else
            dst[i] = '_';
    }
    dst[i] = '\0';
    // TODO: collision handling using a dict
    return string(e->arena, dst);
}

#include <ctype.h>

static enum { ObjectsList, StringLit, CharsLit } array_insides_helper(Emitter* e, Printer* block_printer, Printer* p, Growy* g, const Node* t, Nodes c) {
    if (t->tag == Int_TAG && t->payload.int_type.width == 8) {
        uint8_t* tmp = malloc(sizeof(uint8_t) * c.count);
        bool ends_zero = false;
        for (size_t i = 0; i < c.count; i++) {
            tmp[i] = get_int_literal_value(*resolve_to_int_literal(c.nodes[i]), false);
            if (tmp[i] == 0) {
                if (i == c.count - 1)
                    ends_zero = true;
            }
        }
        bool is_stringy = ends_zero;
        for (size_t i = 0; i < c.count; i++) {
            // ignore the last char in a string
            if (is_stringy && i == c.count - 1)
                break;
            if (isprint(tmp[i]))
                print(p, "%c", tmp[i]);
            else
                print(p, "\\x%02x", tmp[i]);
        }
        free(tmp);
        return is_stringy ? StringLit : CharsLit;
    } else {
        for (size_t i = 0; i < c.count; i++) {
            print(p, to_cvalue(e, emit_value(e, block_printer, c.nodes[i])));
            if (i + 1 < c.count)
                print(p, ", ");
        }
        growy_append_bytes(g, 1, "\0");
        return ObjectsList;
    }
}

static bool has_forward_declarations(CDialect dialect) {
    switch (dialect) {
        case CDialect_C11: return true;
        case CDialect_CUDA: return true;
        case CDialect_GLSL: // no global variable forward declarations in GLSL
        case CDialect_ISPC: // ISPC seems to share this quirk
            return false;
    }
}

static void emit_global_variable_definition(Emitter* emitter, AddressSpace as, String decl_center, const Type* type, bool constant, String init) {
    String prefix = NULL;

    bool is_fs = emitter->compiler_config->specialization.execution_model == EmFragment;
    // GLSL wants 'const' to go on the left to start the declaration, but in C const should go on the right (east const convention)
    switch (emitter->config.dialect) {
        case CDialect_C11: {
            if (as != AsGeneric) warn_print_once(c11_non_generic_as, "warning: standard C does not have address spaces\n");
            prefix = "";
            if (constant)
                decl_center = format_string_arena(emitter->arena->arena, "const %s", decl_center);
            break;
        }
        case CDialect_ISPC:
            // ISPC doesn't really do address space qualifiers.
            prefix = "";
            break;
        case CDialect_CUDA:
            switch (as) {
                case AsPrivate:
                    assert(false);
                    // Note: this requires many hacks.
                    prefix = "__device__ ";
                    decl_center = format_string_arena(emitter->arena->arena, "__shady_private_globals.%s", decl_center);
                    break;
                case AsShared: prefix = "__shared__ "; break;
                case AsGlobal: {
                    if (constant)
                        prefix = "__constant__ ";
                    else
                        prefix = "__device__ __managed__ ";
                    break;
                }
                default: {
                    prefix = format_string_arena(emitter->arena->arena, "/* %s */", get_address_space_name(as));
                    warn_print("warning: address space %s not supported in CUDA for global variables\n", get_address_space_name(as));
                    break;
                }
            }
            break;
        case CDialect_GLSL:
            switch (as) {
                case AsShared: prefix = "shared "; break;
                case AsInput:
                case AsUInput: prefix = emitter->config.glsl_version < 130 ? (is_fs ? "varying " : "attribute ") : "in "; break;
                case AsOutput: prefix = emitter->config.glsl_version < 130 ? "varying " : "out "; break;
                case AsPrivate: prefix = ""; break;
                case AsUniformConstant: prefix = "uniform "; break;
                case AsGlobal: {
                    assert(constant && "Only constants are supported");
                    prefix = "const ";
                    break;
                }
                default: {
                    prefix = format_string_arena(emitter->arena->arena, "/* %s */", get_address_space_name(as));
                    warn_print("warning: address space %s not supported in GLSL for global variables\n", get_address_space_name(as));
                    break;
                }
            }
            break;
    }

    assert(prefix);

    // ISPC wants uniform/varying annotations
    if (emitter->config.dialect == CDialect_ISPC) {
        bool uniform = is_addr_space_uniform(emitter->arena, as);
        if (uniform)
            decl_center = format_string_arena(emitter->arena->arena, "uniform %s", decl_center);
        else
            decl_center = format_string_arena(emitter->arena->arena, "varying %s", decl_center);
    }

    if (init)
        print(emitter->fn_decls, "\n%s%s = %s;", prefix, emit_type(emitter, type, decl_center), init);
    else
        print(emitter->fn_decls, "\n%s%s;", prefix, emit_type(emitter, type, decl_center));

    //if (!has_forward_declarations(emitter->config.dialect) || !init)
    //    return;
    //
    //String declaration = emit_type(emitter, type, decl_center);
    //print(emitter->fn_decls, "\n%s;", declaration);
}

CTerm emit_value(Emitter* emitter, Printer* block_printer, const Node* value) {
    CTerm* found = lookup_existing_term(emitter, value);
    if (found) return *found;

    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_ConstrainedValue_TAG:
        case Value_UntypedNumber_TAG: error("lower me");
        case Param_TAG: error("tried to emit a param: all params should be emitted by their binding abstraction !");
        case Variablez_TAG: error("tried to emit a variable: all variables should be register by their binding let !");
        case Value_IntLiteral_TAG: {
            if (value->payload.int_literal.is_signed)
                emitted = format_string_arena(emitter->arena->arena, "%" PRIi64, value->payload.int_literal.value);
            else
                emitted = format_string_arena(emitter->arena->arena, "%" PRIu64, value->payload.int_literal.value);

            bool is_long = value->payload.int_literal.width == IntTy64;
            bool is_signed = value->payload.int_literal.is_signed;
            if (emitter->config.dialect == CDialect_GLSL && emitter->config.glsl_version >= 130) {
                if (!is_signed)
                    emitted = format_string_arena(emitter->arena->arena, "%sU", emitted);
                if (is_long)
                    emitted = format_string_arena(emitter->arena->arena, "%sL", emitted);
            }

            break;
        }
        case Value_FloatLiteral_TAG: {
            uint64_t v = value->payload.float_literal.value;
            switch (value->payload.float_literal.width) {
                case FloatTy16:
                    assert(false);
                case FloatTy32: {
                    float f;
                    memcpy(&f, &v, sizeof(uint32_t));
                    double d = (double) f;
                    emitted = format_string_arena(emitter->arena->arena, "%#.9gf", d); break;
                }
                case FloatTy64: {
                    double d;
                    memcpy(&d, &v, sizeof(uint64_t));
                    emitted = format_string_arena(emitter->arena->arena, "%.17g", d); break;
                }
            }
            break;
        }
        case Value_True_TAG: return term_from_cvalue("true");
        case Value_False_TAG: return term_from_cvalue("false");
        case Value_Undef_TAG: {
            if (emitter->config.dialect == CDialect_GLSL)
                return emit_value(emitter, block_printer, get_default_zero_value(emitter->arena, value->payload.undef.type));
            String name = unique_name(emitter->arena, "undef");
            emit_global_variable_definition(emitter, AsGlobal, name, value->payload.undef.type, true, NULL);
            emitted = name;
            break;
        }
        case Value_NullPtr_TAG: return term_from_cvalue("NULL");
        case Value_Composite_TAG: {
            const Type* type = value->payload.composite.type;
            Nodes elements = value->payload.composite.contents;

            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            if (type->tag == ArrType_TAG) {
                switch (array_insides_helper(emitter, block_printer, p, g, type, elements)) {
                    case ObjectsList:
                        emitted = growy_data(g);
                        break;
                    case StringLit:
                        emitted = format_string_arena(emitter->arena->arena, "\"%s\"", growy_data(g));
                        break;
                    case CharsLit:
                        emitted = format_string_arena(emitter->arena->arena, "'%s'", growy_data(g));
                        break;
                }
            } else {
                for (size_t i = 0; i < elements.count; i++) {
                    print(p, "%s", to_cvalue(emitter, emit_value(emitter, block_printer, elements.nodes[i])));
                    if (i + 1 < elements.count)
                        print(p, ", ");
                }
                emitted = growy_data(g);
            }
            growy_append_bytes(g, 1, "\0");

            switch (emitter->config.dialect) {
                no_compound_literals:
                case CDialect_ISPC: {
                    // arrays need double the brackets
                    if (type->tag == ArrType_TAG)
                        emitted = format_string_arena(emitter->arena->arena, "{ %s }", emitted);

                    if (block_printer) {
                        String tmp = unique_name(emitter->arena, "composite");
                        print(block_printer, "\n%s = { %s };", emit_type(emitter, value->type, tmp), emitted);
                        emitted = tmp;
                    } else {
                        // this requires us to end up in the initialisation side of a declaration
                        emitted = format_string_arena(emitter->arena->arena, "{ %s }", emitted);
                    }
                    break;
                }
                case CDialect_CUDA:
                case CDialect_C11:
                    // If we're C89 (ew)
                    if (!emitter->config.allow_compound_literals)
                        goto no_compound_literals;
                    emitted = format_string_arena(emitter->arena->arena, "((%s) { %s })", emit_type(emitter, value->type, NULL), emitted);
                    break;
                case CDialect_GLSL:
                    if (type->tag != PackType_TAG)
                        goto no_compound_literals;
                    // GLSL doesn't have compound literals, but it does have constructor syntax for vectors
                    emitted = format_string_arena(emitter->arena->arena, "%s(%s)", emit_type(emitter, value->type, NULL), emitted);
                    break;
            }

            destroy_growy(g);
            destroy_printer(p);
            break;
        }
        case Value_Fill_TAG: error("lower me")
        case Value_StringLiteral_TAG: {
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            String str = value->payload.string_lit.string;
            size_t len = strlen(str);
            for (size_t i = 0; i < len; i++) {
                char c = str[i];
                switch (c) {
                    case '\n': print(p, "\\n");
                        break;
                    default:
                        growy_append_bytes(g, 1, &c);
                }
            }
            growy_append_bytes(g, 1, "\0");

            emitted = format_string_arena(emitter->arena->arena, "\"%s\"", growy_data(g));
            destroy_growy(g);
            destroy_printer(p);
            break;
        }
        case Value_FnAddr_TAG: {
            emitted = legalize_c_identifier(emitter, get_declaration_name(value->payload.fn_addr.fn));
            emitted = format_string_arena(emitter->arena->arena, "(&%s)", emitted);
            break;
        }
        case Value_RefDecl_TAG: {
            const Node* decl = value->payload.ref_decl.decl;
            emit_decl(emitter, decl);

            if (emitter->config.dialect == CDialect_ISPC && decl->tag == GlobalVariable_TAG) {
                if (!is_addr_space_uniform(emitter->arena, decl->payload.global_variable.address_space) && !is_decl_builtin(decl)) {
                    assert(block_printer && "ISPC backend cannot statically refer to a varying variable");
                    return ispc_varying_ptr_helper(emitter, block_printer, decl->type, *lookup_existing_term(emitter, decl));
                }
            }

            return *lookup_existing_term(emitter, decl);
        }
    }

    assert(emitted);
    return term_from_cvalue(emitted);
}

/// hack for ISPC: there is no nice way to get a set of varying pointers (instead of a "pointer to a varying") pointing to a varying global
CTerm ispc_varying_ptr_helper(Emitter* emitter, Printer* block_printer, const Type* ptr_type, CTerm term) {
    String interm = unique_name(emitter->arena, "intermediary_ptr_value");
    const Type* ut = qualified_type_helper(ptr_type, true);
    const Type* vt = qualified_type_helper(ptr_type, false);
    String lhs = emit_type(emitter, vt, interm);
    print(block_printer, "\n%s = ((%s) %s) + programIndex;", lhs, emit_type(emitter, ut, NULL), to_cvalue(emitter, term));
    return term_from_cvalue(interm);
}

void emit_variable_declaration(Emitter* emitter, Printer* block_printer, const Type* t, String variable_name, bool mut, const CTerm* initializer) {
    assert((mut || initializer != NULL) && "unbound results are only allowed when creating a mutable local variable");

    String prefix = "";
    String center = variable_name;

    // add extra qualifiers if immutable
    if (!mut) switch (emitter->config.dialect) {
        case CDialect_ISPC:
            center = format_string_arena(emitter->arena->arena, "const %s", center);
            break;
        case CDialect_C11:
        case CDialect_CUDA:
            center = format_string_arena(emitter->arena->arena, "const %s", center);
            break;
        case CDialect_GLSL:
            if (emitter->config.glsl_version >= 130)
                prefix = "const ";
            break;
    }

    String decl = c_emit_type(emitter, t, center);
    if (initializer)
        print(block_printer, "\n%s%s = %s;", prefix, decl, to_cvalue(emitter, *initializer));
    else
        print(block_printer, "\n%s%s;", prefix, decl);
}

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

    emit_terminator(emitter, p, get_abstraction_body(if_.tail));
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

    emit_terminator(emitter, p, get_abstraction_body(match.tail));
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

    emit_terminator(emitter, p, get_abstraction_body(loop.tail));
}

static void emit_terminator(Emitter* emitter, Printer* block_printer, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Join_TAG: error("this must be lowered away!");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
        case Terminator_BlockYield_TAG:
        case TailCall_TAG: error("TODO");
        case Let_TAG: {
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

            const Node* tail = get_let_tail(terminator);
            assert(tail->tag == Case_TAG);

            Nodes vars = terminator->payload.let.variables;
            assert(vars.count == yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                bool has_result = results[i].value || results[i].var;
                switch (bindings[i]) {
                    case NoBinding: {
                        assert(has_result && "unbound results can't be empty");
                        register_emitted(emitter, vars.nodes[i], results[i]);
                        break;
                    }
                    case LetBinding: {
                        String variable_name = get_value_name_unsafe(vars.nodes[i]);

                        if (!variable_name)
                            variable_name = "";

                        String bind_to = unique_name(emitter->arena, variable_name);

                        const Type* t = yield_types.nodes[i];

                        if (has_result)
                            emit_variable_declaration(emitter, block_printer, t, bind_to, false, &results[i]);
                        else
                            emit_variable_declaration(emitter, block_printer, t, bind_to, false, NULL);

                        register_emitted(emitter, vars.nodes[i], term_from_cvalue(bind_to));
                        break;
                    }
                    default: assert(false);
                }
            }
            emit_terminator(emitter, block_printer, tail->payload.case_.body);

            break;
        }
        case If_TAG: return emit_if(emitter, block_printer, terminator->payload.if_instr);
        case Match_TAG: return emit_match(emitter, block_printer, terminator->payload.match_instr);
        case Loop_TAG: return emit_loop(emitter, block_printer, terminator->payload.loop_instr);
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

void emit_lambda_body_at(Emitter* emitter, Printer* p, const Node* body, const Nodes* bbs) {
    assert(is_terminator(body));
    //print(p, "{");
    indent(p);

    emit_terminator(emitter, p, body);

    if (bbs && bbs->count > 0) {
        assert(emitter->config.dialect != CDialect_GLSL);
        error("TODO");
    }

    deindent(p);
    print(p, "\n");
}

String emit_lambda_body(Emitter* emitter, const Node* body, const Nodes* bbs) {
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);
    emit_lambda_body_at(emitter, p, body, bbs);
    growy_append_bytes(g, 1, (char[]) { 0 });
    return printer_growy_unwrap(p);
}

void emit_decl(Emitter* emitter, const Node* decl) {
    assert(is_declaration(decl));

    CTerm* found = lookup_existing_term(emitter, decl);
    if (found) return;

    CType* found2 = lookup_existing_type(emitter, decl);
    if (found2) return;

    const char* name = legalize_c_identifier(emitter, get_declaration_name(decl));
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    CTerm emit_as;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            String init = NULL;
            if (decl->payload.global_variable.init)
                init = to_cvalue(emitter, emit_value(emitter, NULL, decl->payload.global_variable.init));
            AddressSpace ass = decl->payload.global_variable.address_space;
            if (ass == AsInput || ass == AsOutput)
                init = NULL;

            const GlobalVariable* gvar = &decl->payload.global_variable;
            if (is_decl_builtin(decl)) {
                Builtin b = get_decl_builtin(decl);
                CTerm t = emit_c_builtin(emitter, b);
                register_emitted(emitter, decl, t);
                return;
            }

            if (ass == AsOutput && emitter->compiler_config->specialization.execution_model == EmFragment) {
                int location = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(lookup_annotation(decl, "Location"))), false);
                CTerm t = term_from_cvar(format_string_interned(emitter->arena, "gl_FragData[%d]", location));
                register_emitted(emitter, decl, t);
                return;
            }

            decl_type = decl->payload.global_variable.type;
            // we emit the global variable as a CVar, so we can refer to it's 'address' without explicit ptrs
            emit_as = term_from_cvar(name);
            if ((decl->payload.global_variable.address_space == AsPrivate) && emitter->config.dialect == CDialect_CUDA) {
                if (emitter->use_private_globals) {
                    register_emitted(emitter, decl, term_from_cvar(format_string_arena(emitter->arena->arena, "__shady_private_globals->%s", name)));
                    // HACK
                    return;
                }
                emit_as = term_from_cvar(format_string_interned(emitter->arena, "__shady_thread_local_access(%s)", name));
                if (init)
                    init = format_string_interned(emitter->arena, "__shady_replicate_thread_local(%s)", init);
                register_emitted(emitter, decl, emit_as);
            }
            register_emitted(emitter, decl, emit_as);

            AddressSpace as = decl->payload.global_variable.address_space;
            emit_global_variable_definition(emitter, as, decl_center, decl_type, false, init);
            return;
        }
        case Function_TAG: {
            emit_as = term_from_cvalue(name);
            register_emitted(emitter, decl, emit_as);
            String head = emit_fn_head(emitter, decl->type, name, decl);
            const Node* body = decl->payload.fun.body;
            if (body) {
                for (size_t i = 0; i < decl->payload.fun.params.count; i++) {
                    String param_name;
                    String variable_name = get_value_name_unsafe(decl->payload.fun.params.nodes[i]);
                    param_name = format_string_interned(emitter->arena, "%s_%d", legalize_c_identifier(emitter, variable_name), decl->payload.fun.params.nodes[i]->id);
                    register_emitted(emitter, decl->payload.fun.params.nodes[i], term_from_cvalue(param_name));
                }

                String fn_body = emit_lambda_body(emitter, body, NULL);
                String free_me = fn_body;
                if (emitter->config.dialect == CDialect_ISPC) {
                    // ISPC hack: This compiler (like seemingly all LLVM-based compilers) has broken handling of the execution mask - it fails to generated masked stores for the entry BB of a function that may be called non-uniformingly
                    // therefore we must tell ISPC to please, pretty please, mask everything by branching on what the mask should be
                    fn_body = format_string_arena(emitter->arena->arena, "if ((lanemask() >> programIndex) & 1u) { %s}", fn_body);
                    // I hate everything about this too.
                } else if (emitter->config.dialect == CDialect_CUDA) {
                    if (lookup_annotation(decl, "EntryPoint")) {
                        // fn_body = format_string_arena(emitter->arena->arena, "\n__shady_entry_point_init();%s", fn_body);
                        if (emitter->use_private_globals) {
                            fn_body = format_string_arena(emitter->arena->arena, "\n__shady_PrivateGlobals __shady_private_globals_alloc;\n __shady_PrivateGlobals* __shady_private_globals = &__shady_private_globals_alloc;\n%s", fn_body);
                        }
                        fn_body = format_string_arena(emitter->arena->arena, "\n__shady_prepare_builtins();%s", fn_body);
                    }
                }
                print(emitter->fn_defs, "\n%s { %s }", head, fn_body);
                free_tmp_str(free_me);
            }

            print(emitter->fn_decls, "\n%s;", head);
            return;
        }
        case Constant_TAG: {
            emit_as = term_from_cvalue(name);
            register_emitted(emitter, decl, emit_as);

            const Node* init_value = get_quoted_value(decl->payload.constant.instruction);
            assert(init_value && "TODO: support some measure of constant expressions");
            String init = to_cvalue(emitter, emit_value(emitter, NULL, init_value));
            emit_global_variable_definition(emitter, AsGlobal, decl_center, decl->type, true, init);
            return;
        }
        case NominalType_TAG: {
            CType emitted = name;
            register_emitted_type(emitter, decl, emitted);
            switch (emitter->config.dialect) {
                case CDialect_ISPC:
                default: print(emitter->type_decls, "\ntypedef %s;", emit_type(emitter, decl->payload.nom_type.body, emitted)); break;
                case CDialect_GLSL: emit_nominal_type_body(emitter, format_string_arena(emitter->arena->arena, "struct %s /* nominal */", emitted), decl->payload.nom_type.body); break;
            }
            return;
        }
        default: error("not a decl");
    }
}

void register_emitted(Emitter* emitter, const Node* node, CTerm as) {
    assert(as.value || as.var);
    insert_dict(const Node*, CTerm, emitter->emitted_terms, node, as);
}

void register_emitted_type(Emitter* emitter, const Node* node, String as) {
    insert_dict(const Node*, String, emitter->emitted_types, node, as);
}

CTerm* lookup_existing_term(Emitter* emitter, const Node* node) {
    CTerm* found = find_value_dict(const Node*, CTerm, emitter->emitted_terms, node);
    return found;
}

CType* lookup_existing_type(Emitter* emitter, const Type* node) {
    CType* found = find_value_dict(const Node*, CType, emitter->emitted_types, node);
    return found;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static Module* run_backend_specific_passes(const CompilerConfig* config, CEmitterConfig* econfig, Module* initial_mod) {
    IrArena* initial_arena = initial_mod->arena;
    Module** pmod = &initial_mod;

    // C lacks a nice way to express constants that can be used in type definitions afterwards, so let's just inline them all.
    RUN_PASS(eliminate_constants)
    if (econfig->dialect == CDialect_ISPC) {
        RUN_PASS(lower_workgroups)
    }
    if (econfig->dialect != CDialect_GLSL) {
        RUN_PASS(lower_vec_arr)
    }
    if (config->lower.simt_to_explicit_simd) {
        RUN_PASS(simt2d)
    }
    return *pmod;
}

static String collect_private_globals_in_struct(Emitter* emitter, Module* m) {
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);

    print(p, "typedef struct __shady_PrivateGlobals {\n");
    Nodes decls = get_module_declarations(m);
    size_t count = 0;
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG)
            continue;
        AddressSpace as = decl->payload.global_variable.address_space;
        if (as != AsPrivate)
            continue;
        print(p, "%s;\n", c_emit_type(emitter, decl->payload.global_variable.type, decl->payload.global_variable.name));
        count++;
    }
    print(p, "} __shady_PrivateGlobals;\n");

    if (count == 0) {
        destroy_printer(p);
        destroy_growy(g);
        return NULL;
    }
    return printer_growy_unwrap(p);
}

CEmitterConfig default_c_emitter_config(void) {
    return (CEmitterConfig) {
        .glsl_version = 420,
    };
}

void emit_c(const CompilerConfig* compiler_config, CEmitterConfig config, Module* mod, size_t* output_size, char** output, Module** new_mod) {
    IrArena* initial_arena = get_module_arena(mod);
    mod = run_backend_specific_passes(compiler_config, &config, mod);
    IrArena* arena = get_module_arena(mod);

    Growy* type_decls_g = new_growy();
    Growy* fn_decls_g = new_growy();
    Growy* fn_defs_g = new_growy();

    Emitter emitter = {
        .compiler_config = compiler_config,
        .config = config,
        .arena = arena,
        .type_decls = open_growy_as_printer(type_decls_g),
        .fn_decls = open_growy_as_printer(fn_decls_g),
        .fn_defs = open_growy_as_printer(fn_defs_g),
        .emitted_terms = new_dict(Node*, CTerm, (HashFn) hash_node, (CmpFn) compare_node),
        .emitted_types = new_dict(Node*, String, (HashFn) hash_node, (CmpFn) compare_node),
    };

    // builtins magic (hack) for CUDA
    if (emitter.config.dialect == CDialect_CUDA) {
        emitter.total_workgroup_size = emitter.arena->config.specializations.workgroup_size[0];
        emitter.total_workgroup_size *= emitter.arena->config.specializations.workgroup_size[1];
        emitter.total_workgroup_size *= emitter.arena->config.specializations.workgroup_size[2];
        print(emitter.type_decls, "\ntypedef %s;\n", emit_type(&emitter, arr_type(arena, (ArrType) {
                .size = int32_literal(arena, 3),
                .element_type = uint32_type(arena)
        }), "uvec3"));
        print(emitter.fn_defs, shady_cuda_builtins_src);

        String private_globals = collect_private_globals_in_struct(&emitter, mod);
        if (private_globals) {
            emitter.use_private_globals = true;
            print(emitter.type_decls, private_globals);
            free((void*)private_globals);
        }
    }

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++)
        emit_decl(&emitter, decls.nodes[i]);

    destroy_printer(emitter.type_decls);
    destroy_printer(emitter.fn_decls);
    destroy_printer(emitter.fn_defs);

    Growy* final = new_growy();
    Printer* finalp = open_growy_as_printer(final);

    if (emitter.config.dialect == CDialect_GLSL) {
        print(finalp, "#version %d\n", emitter.config.glsl_version);
    }

    print(finalp, "/* file generated by shady */\n");

    switch (emitter.config.dialect) {
        case CDialect_ISPC:
            break;
        case CDialect_CUDA: {
            print(finalp, "#define __shady_workgroup_size %d\n", emitter.total_workgroup_size);
            print(finalp, "#define __shady_replicate_thread_local(v) { ");
            for (size_t i = 0; i < emitter.total_workgroup_size; i++)
                print(finalp, "v, ");
            print(finalp, "}\n");
            print(finalp, shady_cuda_prelude_src);
            break;
        }
        case CDialect_C11:
            print(finalp, "\n#include <stdbool.h>");
            print(finalp, "\n#include <stdint.h>");
            print(finalp, "\n#include <stddef.h>");
            print(finalp, "\n#include <stdio.h>");
            print(finalp, "\n#include <math.h>");
            break;
        case CDialect_GLSL:
            if (emitter.need_64b_ext)
                print(finalp, "#extension GL_ARB_gpu_shader_int64: require\n");
            print(finalp, "#define ubyte uint\n");
            print(finalp, "#define uchar uint\n");
            print(finalp, "#define ulong uint\n");
            if (emitter.config.glsl_version <= 120)
                print(finalp, shady_glsl_120_polyfills_src);
            break;
    }

    print(finalp, "\n/* types: */\n");
    growy_append_bytes(final, growy_size(type_decls_g), growy_data(type_decls_g));

    print(finalp, "\n/* declarations: */\n");
    growy_append_bytes(final, growy_size(fn_decls_g), growy_data(fn_decls_g));

    print(finalp, "\n/* definitions: */\n");
    growy_append_bytes(final, growy_size(fn_defs_g), growy_data(fn_defs_g));

    print(finalp, "\n");
    print(finalp, "\n");
    print(finalp, "\n");
    growy_append_bytes(final, 1, "\0");

    destroy_growy(type_decls_g);
    destroy_growy(fn_decls_g);
    destroy_growy(fn_defs_g);

    destroy_dict(emitter.emitted_types);
    destroy_dict(emitter.emitted_terms);

    *output_size = growy_size(final) - 1;
    *output = growy_deconstruct(final);
    destroy_printer(finalp);

    if (new_mod)
        *new_mod = mod;
    else if (initial_arena != arena)
        destroy_ir_arena(arena);
}
