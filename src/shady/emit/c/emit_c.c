#include "emit_c.h"

#include "portability.h"
#include "dict.h"
#include "log.h"

#include "../../type.h"
#include "../../ir_private.h"

#include <assert.h>
#include <stdlib.h>

#pragma GCC diagnostic error "-Wswitch"

static String emit_decl(Emitter* emitter, const Node* decl);
static void emit_terminator(Emitter* emitter, Printer* p, const Node* terminator);

String emit_fn_head(Emitter* emitter, const Node* fn) {
    assert(fn->tag == Lambda_TAG);
    Nodes dom = fn->payload.lam.params;
    Nodes codom = fn->payload.lam.return_types;

    Growy* paramg = new_growy();
    Printer* paramp = open_growy_as_printer(paramg);
    if (dom.count == 0)
        print(paramp, "void");
    else for (size_t i = 0; i < dom.count; i++) {
        print(paramp, emit_type(emitter, dom.nodes[i]->type, format_string(emitter->arena, "%s_%d", dom.nodes[i]->payload.var.name, dom.nodes[i]->payload.var.id)));
        if (i + 1 < dom.count) {
            print(paramp, ", ");
        }
    }
    growy_append_bytes(paramg, 1, (char[]) { 0 });
    const char* parameters = printer_growy_unwrap(paramp);
    String center = format_string(emitter->arena, "%s(%s)", fn->payload.lam.name, parameters);
    free_tmp_str(parameters);

    return emit_type(emitter, wrap_multiple_yield_types(emitter->arena, codom), center);
}

#include <ctype.h>

static enum { ObjectsList, StringLit, CharsLit } array_insides_helper(Emitter* e, Printer* p, Growy* g, const Node* array) {
    const Type* t = array->payload.arr_lit.element_type;
    Nodes c = array->payload.arr_lit.contents;
    if (t->tag == Int_TAG && t->payload.int_type.width == 8) {
        uint8_t* tmp = malloc(sizeof(uint8_t) * c.count);
        bool ends_zero = false;
        for (size_t i = 0; i < c.count; i++) {
            tmp[i] = extract_int_literal_value(c.nodes[i], false);
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
            print(p, emit_value(e, c.nodes[i]));
            if (i + 1 < c.count)
                print(p, ", ");
        }
        growy_append_bytes(g, 1, "\0");
        return ObjectsList;
    }
}

String emit_value(Emitter* emitter, const Node* value) {
    String* found = find_value_dict(const Node*, String, emitter->emitted, value);
    if (found)
        return *found;

    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_Unbound_TAG:
        case Value_UntypedNumber_TAG: error("lower me");
        case Value_Variable_TAG: error("variables need to be emitted beforehand");
        case Value_IntLiteral_TAG: emitted = format_string(emitter->arena, "%d", value->payload.int_literal.value_u64); break;
        case Value_True_TAG: return "true";
        case Value_False_TAG: return "false";
        case Value_Tuple_TAG: break;
        case Value_StringLiteral_TAG: break;
        case Value_ArrayLiteral_TAG: {
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);
            switch (array_insides_helper(emitter, p, g, value)) {
                case ObjectsList:
                    emitted = format_string(emitter->arena, "((%s) { %s })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
                case StringLit:
                    emitted = format_string(emitter->arena, "((%s) { \"s\" })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
                case CharsLit:
                    emitted = format_string(emitter->arena, "((%s) { '%s' })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
            }
            growy_destroy(g);
            destroy_printer(p);
            break;
        }
        case Value_FnAddr_TAG: {
            emitted = get_decl_name(value->payload.fn_addr.fn);
            emitted = format_string(emitter->arena, "&%s", emitted);
            break;
        }
        case Value_RefDecl_TAG: {
            emitted = emit_decl(emitter, value->payload.ref_decl.decl);
            break;
        }
    }

    assert(emitted);
    return emitted;
}

static void emit_cf_destination(Emitter* emitter, Printer* p, const Node* destination, size_t num_args, const String args[]) {
    assert(destination->tag == Lambda_TAG);
    if (destination->payload.lam.tier == FnTier_BasicBlock) {
        assert(num_args == 0 && "TODO phis");
        print(p, "\ngoto %s;", emit_decl(emitter, destination));
    } else {
        assert(destination->payload.lam.tier == FnTier_Lambda);
        for (size_t i = 0; i < num_args; i++)
            insert_dict(const Node*, String, emitter->emitted, destination->payload.lam.params.nodes[i], args[i]);
        emit_terminator(emitter, p, destination->payload.lam.body);
    }
}

static void emit_terminator(Emitter* emitter, Printer* p, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Terminator_Join_TAG:
        case Terminator_Branch_TAG: error("this must be lowered away!");
        case Terminator_TailCall_TAG: error("TODO");
        case Terminator_Let_TAG: {
            const Node* instruction = terminator->payload.let.instruction;
            // we declare N local variables in order to store the result of the instruction
            Nodes yield_types = unwrap_multiple_yield_types(emitter->arena, instruction->type);
            const Nodes tail_params = terminator->payload.let.tail->payload.lam.params;
            assert(tail_params.count == yield_types.count);
            LARRAY(String, output_names, yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                output_names[i] = format_string(emitter->arena, "%s_%d", tail_params.nodes[i]->payload.var.name, fresh_id(emitter->arena));
                print(p, "\n%s;", c_emit_type(emitter, yield_types.nodes[i], output_names[i]));
            }
            emit_instruction(emitter, p, instruction, strings(emitter->arena, yield_types.count, output_names));
            emit_cf_destination(emitter, p, terminator->payload.let.tail, yield_types.count, output_names);
            break;
        }
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.values;
            if (args.count == 0) {
                print(p, "\nreturn;");
            } else if (args.count == 1) {
                print(p, "\nreturn %s;", emit_value(emitter, args.nodes[0]));
            } else {
                String packed = unique_name(emitter->arena, "pack_return");
                emit_pack_code(p, emit_values(emitter, args), packed);
                print(p, "\nreturn %s;", packed);
            }
            break;
        }
        case Terminator_MergeConstruct_TAG: {
            Nodes args = terminator->payload.merge_construct.args;
            Phis phis;
            switch (terminator->payload.merge_construct.construct) {
                case Selection: phis = emitter->phis.selection; break;
                case Continue:  phis = emitter->phis.loop_continue; break;
                case Break:     phis = emitter->phis.loop_break; break;
            }
            assert(phis.count == args.count);
            for (size_t i = 0; i < phis.count; i++)
                print(p, "\n%s = %s;", phis.strings[i], emit_value(emitter, args.nodes[i]));

            switch (terminator->payload.merge_construct.construct) {
                case Selection: break;
                case Continue:  print(p, "\ncontinue;"); break;
                case Break:     print(p, "\nbreak;"); break;
            }
            break;
        }
        case Terminator_Unreachable_TAG: {
            assert(emitter->config.dialect == C);
            print(p, "\n__builtin_unreachable();");
            break;
        }
    }
}

void emit_lambda_body_at(Emitter* emitter, Printer* p, const Node* body, const Nodes* bbs) {
    assert(is_terminator(body));
    print(p, "{");
    indent(p);

    emit_terminator(emitter, p, body);

    if (bbs && bbs->count > 0) {
        assert(emitter->config.dialect != GLSL);
        error("TODO");
    }

    deindent(p);
    print(p, "\n}");
}

String emit_lambda_body(Emitter* emitter, const Node* body, const Nodes* bbs) {
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);
    emit_lambda_body_at(emitter, p, body, bbs);
    growy_append_bytes(g, 1, (char[]) { 0 });
    return printer_growy_unwrap(p);
}

static String emit_decl(Emitter* emitter, const Node* decl) {
    assert(is_declaration(decl));

    String* found = find_value_dict(const Node*, String, emitter->emitted, decl);
    if (found)
        return *found;

    const char* name = get_decl_name(decl);
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    const char* emit_as = NULL;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            decl_type = decl->payload.global_variable.type;
            // users of the global variable are actually using its address
            emit_as = format_string(emitter->arena, "(&%s)", name);
            register_emitted(emitter, decl, emit_as);
            if (decl->payload.global_variable.init)
                print(emitter->fn_defs, "\n%s = %s;", emit_type(emitter, decl_type, decl_center), emit_value(emitter, decl->payload.global_variable.init));
            break;
        }
        case Lambda_TAG: {
            emit_as = name;
            register_emitted(emitter, decl, emit_as);
            const Node* body = decl->payload.lam.body;
            if (body) {
                for (size_t i = 0; i < decl->payload.lam.params.count; i++) {
                    const char* param_name = format_string(emitter->arena, "%s_%d", decl->payload.lam.params.nodes[i]->payload.var.name, decl->payload.lam.params.nodes[i]->payload.var.id);
                    register_emitted(emitter, decl->payload.lam.params.nodes[i], param_name);
                }

                String fn_body = emit_lambda_body(emitter, body, NULL);
                print(emitter->fn_defs, "\n%s %s", emit_fn_head(emitter, decl), fn_body);
                free_tmp_str(fn_body);
            }
            break;
        }
        case Constant_TAG: {
            emit_as = name;
            register_emitted(emitter, decl, emit_as);
            decl_center = format_string(emitter->arena, "const %s", decl_center);
            print(emitter->fn_defs, "\n%s = %s;", emit_type(emitter, decl->type, decl_center), emit_value(emitter, decl->payload.constant.value));
            break;
        }
        default: error("not a decl");
    }

    String declaration = emit_type(emitter, decl_type, decl_center);
    print(emitter->fn_decls, "\n%s;", declaration);
    return emit_as;
}

void register_emitted(Emitter* emitter, const Node* node, String as) {
    insert_dict(const Node*, String, emitter->emitted, node, as);
}

void register_emitted_list(Emitter* emitter, Nodes nodes, Strings as) {
    assert(nodes.count == as.count);
    for (size_t i = 0; i < nodes.count; i++)
        register_emitted(emitter, nodes.nodes[i], as.strings[i]);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void emit_c(CompilerConfig* config, IrArena* arena, const Node* root_node, size_t* output_size, char** output) {
    Growy* type_decls_g = new_growy();
    Growy* fn_decls_g = new_growy();
    Growy* fn_defs_g = new_growy();

    Emitter emitter = {
        .config = {
            .config = config,
            .dialect = C,
        },
        .arena = arena,
        .type_decls = open_growy_as_printer(type_decls_g),
        .fn_decls = open_growy_as_printer(fn_decls_g),
        .fn_defs = open_growy_as_printer(fn_defs_g),
        .emitted = new_dict(Node*, String, (HashFn) hash_node, (CmpFn) compare_node)
    };

    assert(root_node->tag == Root_TAG);
    Nodes decls = root_node->payload.root.declarations;
    for (size_t i = 0; i < decls.count; i++)
        emit_decl(&emitter, decls.nodes[i]);

    destroy_printer(emitter.type_decls);
    destroy_printer(emitter.fn_decls);
    destroy_printer(emitter.fn_defs);

    Growy* final = new_growy();
    Printer* finalp = open_growy_as_printer(final);

    print(finalp, "/* file generated by shady */\n");

    if (emitter.config.dialect == C) {
        print(finalp, "\n#include <stdbool.h>");
        print(finalp, "\n#include <stdint.h>");
        print(finalp, "\n#include <stddef.h>");
    }

    print(finalp, "\n/* types: */\n");
    growy_append_bytes(final, growy_size(type_decls_g), growy_data(type_decls_g));

    print(finalp, "\n/* declarations: */\n");
    growy_append_bytes(final, growy_size(fn_decls_g), growy_data(fn_decls_g));

    print(finalp, "\n/* definitions: */\n");
    growy_append_bytes(final, growy_size(fn_defs_g), growy_data(fn_defs_g));

    print(finalp, "\n");

    growy_destroy(type_decls_g);
    growy_destroy(fn_decls_g);
    growy_destroy(fn_defs_g);

    destroy_dict(emitter.emitted);

    *output_size = growy_size(final);
    *output = growy_deconstruct(final);
    destroy_printer(finalp);
}
