#include "emit_c.h"

#include "portability.h"
#include "dict.h"
#include "log.h"

#include "../../type.h"
#include "../../ir_private.h"
#include "../../compile.h"

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
        return format_string(e->arena, "(&%s)", term.var);
    assert(false);
}

CAddr deref_term(Emitter* e, CTerm term) {
    if (term.value)
        return format_string(e->arena, "(*%s)", term.value);
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
            tmp[i] = get_int_literal_value(c.nodes[i], false);
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
        case C: return true;
        case GLSL: // no global variable forward declarations in GLSL
        case ISPC: // ISPC seems to share this quirk
            return false;
    }
}

static void emit_global_variable_definition(Emitter* emitter, String prefix, String decl_center, const Type* type, bool uniform, bool constant, String init) {
    // GLSL wants 'const' to go on the left to start the declaration, but in C const should go on the right (east const convention)
    switch (emitter->config.dialect) {
        // ISPC defaults to varying, even for constants... yuck
        case ISPC:
            if (uniform)
                decl_center = format_string(emitter->arena, "uniform %s", decl_center);
            else
                decl_center = format_string(emitter->arena, "varying %s", decl_center);
            break;
        case C:
            if (constant)
                decl_center = format_string(emitter->arena, "const %s", decl_center);
            break;
        case GLSL:
            if (constant)
                prefix = format_string(emitter->arena, "%s %s", "const", prefix);
            break;
    }

    if (init)
        print(emitter->fn_defs, "\n%s%s = %s;", prefix, emit_type(emitter, type, decl_center), init);
    else
        print(emitter->fn_defs, "\n%s%s;", prefix, emit_type(emitter, type, decl_center));

    if (!has_forward_declarations(emitter->config.dialect) || !init)
        return;

    String declaration = emit_type(emitter, type, decl_center);
    print(emitter->fn_decls, "\n%s;", declaration);
}

CTerm emit_value(Emitter* emitter, Printer* block_printer, const Node* value) {
    CTerm* found = lookup_existing_term(emitter, value);
    if (found) return *found;

    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_AntiQuote_TAG:
        case Value_ConstrainedValue_TAG:
        case Value_UntypedNumber_TAG: error("lower me");
        case Value_Variable_TAG: error("variables need to be emitted beforehand");
        case Value_IntLiteral_TAG: {
            if (value->payload.int_literal.is_signed)
                emitted = format_string(emitter->arena, "%" PRIi64, value->payload.int_literal.value);
            else
                emitted = format_string(emitter->arena, "%" PRIu64, value->payload.int_literal.value);
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
                    emitted = format_string(emitter->arena, "%.9g", d); break;
                }
                case FloatTy64: {
                    double d;
                    memcpy(&d, &v, sizeof(uint64_t));
                    emitted = format_string(emitter->arena, "%.17g", d); break;
                }
            }
            break;
        }
        case Value_True_TAG: return term_from_cvalue("true");
        case Value_False_TAG: return term_from_cvalue("false");
        case Value_Undef_TAG: {
            String name = unique_name(emitter->arena, "undef");
            emit_global_variable_definition(emitter, "", name, value->payload.undef.type, true, true, NULL);
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
                        emitted = format_string(emitter->arena, "\"%s\"", growy_data(g));
                        break;
                    case CharsLit:
                        emitted = format_string(emitter->arena, "'%s'", growy_data(g));
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
                case ISPC: {
                    // arrays need double the brackets
                    if (type->tag == ArrType_TAG)
                        emitted = format_string(emitter->arena, "{ %s }", emitted);

                    if (block_printer) {
                        String tmp = unique_name(emitter->arena, "composite");
                        print(block_printer, "\n%s = { %s };", emit_type(emitter, value->type, tmp), emitted);
                        emitted = tmp;
                    } else {
                        // this requires us to end up in the initialisation side of a declaration
                        emitted = format_string(emitter->arena, "{ %s }", emitted);
                    }
                    break;
                }
                case C:
                    // If we're C89 (ew)
                    if (!emitter->config.allow_compound_literals)
                        goto no_compound_literals;
                    emitted = format_string(emitter->arena, "((%s) { %s })", emit_type(emitter, value->type, NULL), emitted);
                    break;
                case GLSL:
                    if (type->tag != PackType_TAG)
                        goto no_compound_literals;
                    // GLSL doesn't have compound literals, but it does have constructor syntax for vectors
                    emitted = format_string(emitter->arena, "%s(%s)", type, emitted);
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

            emitted = format_string(emitter->arena, "\"%s\"", growy_data(g));
            destroy_growy(g);
            destroy_printer(p);
            break;
        }
        case Value_FnAddr_TAG: {
            emitted = legalize_c_identifier(emitter, get_decl_name(value->payload.fn_addr.fn));
            emitted = format_string(emitter->arena, "(&%s)", emitted);
            break;
        }
        case Value_RefDecl_TAG: {
            const Node* decl = value->payload.ref_decl.decl;
            emit_decl(emitter, decl);

            if (emitter->config.dialect == ISPC && decl->tag == GlobalVariable_TAG) {
                if (!is_addr_space_uniform(emitter->arena, decl->payload.global_variable.address_space)) {
                    assert(block_printer && "ISPC backend cannot statically refer to a varying global variable");
                    // hack for ISPC: there is no nice way to get a set of varying pointers (instead of a "pointer to a varying") pointing to a varying global
                    String interm = unique_name(emitter->arena, "intermediary_ptr_value");
                    const Type* ut = qualified_type_helper(decl->type, true);
                    const Type* vt = qualified_type_helper(decl->type, false);
                    String lhs = emit_type(emitter, vt, interm);
                    print(block_printer, "\n%s = ((%s) %s) + programIndex;", lhs, emit_type(emitter, ut, NULL), to_cvalue(emitter, *lookup_existing_term(emitter, decl)));
                    return term_from_cvalue(interm);
                }
            }

            return *lookup_existing_term(emitter, decl);
        }
    }

    assert(emitted);
    return term_from_cvalue(emitted);
}

static void emit_terminator(Emitter* emitter, Printer* block_printer, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case LetMut_TAG:
        case Join_TAG: error("this must be lowered away!");
        case Jump_TAG:
        case Branch_TAG:
        case Switch_TAG:
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
            assert(tail->tag == AnonLambda_TAG);

            const Nodes tail_params = tail->payload.anon_lam.params;
            assert(tail_params.count == yield_types.count);
            for (size_t i = 0; i < yield_types.count; i++) {
                bool mut = false;
                bool has_result = results[i].value || results[i].var;
                switch (bindings[i]) {
                    case NoBinding: {
                        assert(has_result && "unbound results can't be empty");
                        register_emitted(emitter, tail_params.nodes[i], results[i]);
                        break;
                    }
                    case LetMutBinding: mut = true;
                    case LetBinding: {
                        assert((mut || has_result) && "unbound results are only allowed when creating a mutable local variable");
                        String bind_to = format_string(emitter->arena, "%s_%d", legalize_c_identifier(emitter, tail_params.nodes[i]->payload.var.name), fresh_id(emitter->arena));

                        String prefix = "";
                        String center = bind_to;

                        // add extra qualifiers if immutable
                        if (!mut) switch (emitter->config.dialect) {
                            case ISPC:
                                center = format_string(emitter->arena, "const %s", bind_to);
                                break;
                            case C:
                                prefix = "register ";
                                center = format_string(emitter->arena, "const %s", bind_to);
                                break;
                            case GLSL:
                                prefix = "const ";
                                break;
                        }

                        const Type* t = yield_types.nodes[i];
                        if (mut)
                            t = get_pointee_type(emitter->arena, t);

                        String decl = c_emit_type(emitter, t, center);
                        if (has_result)
                            print(block_printer, "\n%s%s = %s;", prefix, decl, to_cvalue(emitter, results[i]));
                        else
                            print(block_printer, "\n%s%s;", prefix, decl);

                        if (mut)
                            register_emitted(emitter, tail_params.nodes[i], term_from_cvar(bind_to));
                        else
                            register_emitted(emitter, tail_params.nodes[i], term_from_cvalue(bind_to));
                        break;
                    }
                    default: assert(false);
                }
            }
            emit_terminator(emitter, block_printer, tail->payload.anon_lam.body);

            break;
        }
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
        case Yield_TAG: {
            Nodes args = terminator->payload.yield.args;
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
                case C:
                    print(block_printer, "\n__builtin_unreachable();");
                    break;
                case ISPC:
                    print(block_printer, "\nassert(false);");
                    break;
                case GLSL:
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
        assert(emitter->config.dialect != GLSL);
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

    const char* name = legalize_c_identifier(emitter, get_decl_name(decl));
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    CTerm emit_as;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            String init = NULL;
            if (decl->payload.global_variable.init)
                init = to_cvalue(emitter, emit_value(emitter, NULL, decl->payload.global_variable.init));

            const GlobalVariable* gvar = &decl->payload.global_variable;
            Builtin b = BuiltinsCount;
            for (size_t i = 0; i < gvar->annotations.count; i++) {
                const Node* a = gvar->annotations.nodes[i];
                assert(is_annotation(a));
                if (strcmp(get_annotation_name(a), "Builtin") == 0) {
                    String builtin_name = get_annotation_string_payload(a);
                    assert(builtin_name);
                    b = get_builtin_by_name(builtin_name);
                    assert(b != BuiltinsCount);
                    name = builtin_name;
                    decl_center = builtin_name;
                    //register_emitted(emitter, decl, emit_c_builtin(emitter, p, b));
                    //return;
                }
            }

            if (b == BuiltinSubgroupLocalInvocationId)
                init = "programIndex";

            decl_type = decl->payload.global_variable.type;
            // we emit the global variable as a CVar, so we can refer to it's 'address' without explicit ptrs
            emit_as = term_from_cvar(name);

            bool uniform = is_addr_space_uniform(emitter->arena, decl->payload.global_variable.address_space);

            String address_space_prefix = NULL;
            switch (decl->payload.global_variable.address_space) {
                case AsGeneric:
                case AsSubgroupLogical:
                case AsSubgroupPhysical:
                    switch (emitter->config.dialect) {
                        case C:
                        case GLSL:
                            warn_print("C and GLSL do not have a 'subgroup' level addressing space, using shared instead");
                            address_space_prefix = "shared ";
                            break;
                        case ISPC:
                            address_space_prefix = "";
                            break;
                    }
                    break;
                case AsPrivatePhysical:
                case AsPrivateLogical:
                    address_space_prefix = "";
                case AsGlobalLogical:
                case AsGlobalPhysical:
                    address_space_prefix = "";
                    break;
                case AsSharedPhysical:
                case AsSharedLogical:
                    switch (emitter->config.dialect) {
                        case C:
                            break;
                        case GLSL:
                            address_space_prefix = "shared ";
                            break;
                        case ISPC:
                            // ISPC doesn't really know what "shared" is
                            break;
                    }
                    break;
                case AsExternal:
                    address_space_prefix = "extern ";
                    break;
                case AsProgramCode:
                case AsInput:
                case AsUInput:
                case AsOutput:
                    break;
                case AsGLUniformBufferObject:
                case AsGLShaderStorageBufferObject:
                case AsSPVFunctionLogical:
                case AsVKPushConstant:
                    break; // error("These only make sense for SPIR-V !")
                case NumAddressSpaces: error("");
            }

            if (!address_space_prefix) {
                warn_print("No known address space prefix for as %d, this might produce broken code\n", decl->payload.global_variable.address_space);
                address_space_prefix = "";
            }

            register_emitted(emitter, decl, emit_as);

            emit_global_variable_definition(emitter, address_space_prefix, decl_center, decl_type, uniform, false, init);
            return;
        }
        case Function_TAG: {
            emit_as = term_from_cvalue(name);
            register_emitted(emitter, decl, emit_as);
            String head = emit_fn_head(emitter, decl->type, name, decl);
            const Node* body = decl->payload.fun.body;
            if (body) {
                for (size_t i = 0; i < decl->payload.fun.params.count; i++) {
                    const char* param_name = format_string(emitter->arena, "%s_%d", legalize_c_identifier(emitter, decl->payload.fun.params.nodes[i]->payload.var.name), decl->payload.fun.params.nodes[i]->payload.var.id);
                    register_emitted(emitter, decl->payload.fun.params.nodes[i], term_from_cvalue(param_name));
                }

                String fn_body = emit_lambda_body(emitter, body, NULL);
                String free_me = fn_body;
                if (emitter->config.dialect == ISPC) {
                    // ISPC hack: This compiler (like seemingly all LLVM-based compilers) has broken handling of the execution mask - it fails to generated masked stores for the entry BB of a function that may be called non-uniformingly
                    // therefore we must tell ISPC to please, pretty please, mask everything by branching on what the mask should be
                    fn_body = format_string(emitter->arena, "if ((lanemask() >> programIndex) & 1u) { %s}", fn_body);
                    // I hate everything about this too.
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

            String init = to_cvalue(emitter, emit_value(emitter, NULL, decl->payload.constant.value));
            emit_global_variable_definition(emitter, "", decl_center, decl->type, true, true, init);
            return;
        }
        case NominalType_TAG: {
            CType emitted = name;
            register_emitted_type(emitter, decl, emitted);
            switch (emitter->config.dialect) {
                case ISPC:
                case C: print(emitter->type_decls, "\ntypedef %s;", emit_type(emitter, decl->payload.nom_type.body, emitted)); break;
                case GLSL: emit_nominal_type_body(emitter, format_string(emitter->arena, "struct %s /* nominal */", emitted), decl->payload.nom_type.body); break;
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

static Module* run_backend_specific_passes(SHADY_UNUSED CEmitterConfig* econfig, Module* mod) {
    CompilerConfig* config = econfig->config;
    IrArena* old_arena = get_module_arena(mod);
    ArenaConfig aconfig = old_arena->config;
    Module* old_mod;
    IrArena* tmp_arena = NULL;
    if (econfig->dialect == ISPC) {
        RUN_PASS(lower_workgroups)
    }
    if (econfig->dialect != GLSL) {
        aconfig.validate_builtin_types = false;
        RUN_PASS(lower_vec_arr)
    }
    if (config->lower.simt_to_explicit_simd) {
        aconfig.is_simt = false;
        RUN_PASS(simt2d)
    }
    // C lacks a nice way to express constants that can be used in type definitions afterwards, so let's just inline them all.
    RUN_PASS(eliminate_constants)
    return mod;
}

void emit_c(CEmitterConfig config, Module* mod, size_t* output_size, char** output, Module** new_mod) {
    IrArena* initial_arena = get_module_arena(mod);
    mod = run_backend_specific_passes(&config, mod);
    IrArena* arena = get_module_arena(mod);

    Growy* type_decls_g = new_growy();
    Growy* fn_decls_g = new_growy();
    Growy* fn_defs_g = new_growy();

    Emitter emitter = {
        .config = config,
        .arena = arena,
        .type_decls = open_growy_as_printer(type_decls_g),
        .fn_decls = open_growy_as_printer(fn_decls_g),
        .fn_defs = open_growy_as_printer(fn_defs_g),
        .emitted_terms = new_dict(Node*, CTerm, (HashFn) hash_node, (CmpFn) compare_node),
        .emitted_types = new_dict(Node*, String, (HashFn) hash_node, (CmpFn) compare_node),
    };

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++)
        emit_decl(&emitter, decls.nodes[i]);

    destroy_printer(emitter.type_decls);
    destroy_printer(emitter.fn_decls);
    destroy_printer(emitter.fn_defs);

    Growy* final = new_growy();
    Printer* finalp = open_growy_as_printer(final);

    if (emitter.config.dialect == GLSL) {
        print(finalp, "#version 420\n");
    }

    print(finalp, "/* file generated by shady */\n");

    switch (emitter.config.dialect) {
        case ISPC:
            break;
        case C:
            print(finalp, "\n#include <stdbool.h>");
            print(finalp, "\n#include <stdint.h>");
            print(finalp, "\n#include <stddef.h>");
            print(finalp, "\n#include <stdio.h>");
            print(finalp, "\n#include <math.h>");
            break;
        case GLSL:
            print(finalp, "#extension GL_ARB_compute_shader: require\n");
            print(finalp, "#define ubyte uint\n");
            print(finalp, "#define uchar uint\n");
            print(finalp, "#define ulong uint\n");
            break;
    }

    print(finalp, "\n/* types: */\n");
    growy_append_bytes(final, growy_size(type_decls_g), growy_data(type_decls_g));

    print(finalp, "\n/* declarations: */\n");
    growy_append_bytes(final, growy_size(fn_decls_g), growy_data(fn_decls_g));

    print(finalp, "\n/* definitions: */\n");
    growy_append_bytes(final, growy_size(fn_defs_g), growy_data(fn_defs_g));

    print(finalp, "\n");

    destroy_growy(type_decls_g);
    destroy_growy(fn_decls_g);
    destroy_growy(fn_defs_g);

    destroy_dict(emitter.emitted_types);
    destroy_dict(emitter.emitted_terms);

    *output_size = growy_size(final);
    *output = growy_deconstruct(final);
    destroy_printer(finalp);

    if (new_mod)
        *new_mod = mod;
    else if (initial_arena != arena)
        destroy_ir_arena(arena);
}
