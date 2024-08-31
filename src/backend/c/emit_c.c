#include "emit_c.h"

#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/compile.h"
#include "../shady/transform/ir_gen_helpers.h"
#include "../shady/analysis/cfg.h"
#include "../shady/analysis/scheduler.h"

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

#pragma GCC diagnostic error "-Wswitch"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void register_emitted(Emitter* emitter, FnEmitter* fn, const Node* node, CTerm as) {
    //assert(as.value || as.var);
    insert_dict(const Node*, CTerm, fn ? fn->emitted_terms : emitter->emitted_terms, node, as);
}

CTerm* lookup_existing_term(Emitter* emitter, FnEmitter* fn, const Node* node) {
    CTerm* found = NULL;
    if (fn)
        found = find_value_dict(const Node*, CTerm, fn->emitted_terms, node);
    if (!found)
        found = find_value_dict(const Node*, CTerm, emitter->emitted_terms, node);
    return found;
}

void register_emitted_type(Emitter* emitter, const Node* node, String as) {
    insert_dict(const Node*, String, emitter->emitted_types, node, as);
}

CType* lookup_existing_type(Emitter* emitter, const Type* node) {
    CType* found = find_value_dict(const Node*, CType, emitter->emitted_types, node);
    return found;
}

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
static bool c_is_legal_identifier_char(char c) {
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

String c_legalize_identifier(Emitter* e, String src) {
    if (!src)
        return "unnamed";
    size_t len = strlen(src);
    LARRAY(char, dst, len + 1);
    size_t i;
    for (i = 0; i < len; i++) {
        char c = src[i];
        if (c_is_legal_identifier_char(c))
            dst[i] = c;
        else
            dst[i] = '_';
    }
    dst[i] = '\0';
    // TODO: collision handling using a dict
    return string(e->arena, dst);
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

/// hack for ISPC: there is no nice way to get a set of varying pointers (instead of a "pointer to a varying") pointing to a varying global
CTerm ispc_varying_ptr_helper(Emitter* emitter, Printer* block_printer, const Type* ptr_type, CTerm term) {
    String interm = unique_name(emitter->arena, "intermediary_ptr_value");
    assert(ptr_type->tag == PtrType_TAG);
    const Type* ut = qualified_type_helper(ptr_type, true);
    const Type* vt = qualified_type_helper(ptr_type, false);
    String lhs = c_emit_type(emitter, vt, interm);
    print(block_printer, "\n%s = ((%s) %s) + programIndex;", lhs, c_emit_type(emitter, ut, NULL), to_cvalue(emitter, term));
    return term_from_cvalue(interm);
}

void c_emit_variable_declaration(Emitter* emitter, Printer* block_printer, const Type* t, String variable_name, bool mut, const CTerm* initializer) {
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

void c_emit_pack_code(Printer* p, Strings src, String dst) {
    for (size_t i = 0; i < src.count; i++) {
        print(p, "\n%s->_%d = %s", dst, src.strings[i], i);
    }
}

void c_emit_unpack_code(Printer* p, String src, Strings dst) {
    for (size_t i = 0; i < dst.count; i++) {
        print(p, "\n%s = %s->_%d", dst.strings[i], src, i);
    }
}

void c_emit_global_variable_definition(Emitter* emitter, AddressSpace as, String name, const Type* type, bool constant, String init) {
    String prefix = NULL;

    bool is_fs = emitter->compiler_config->specialization.execution_model == EmFragment;
    // GLSL wants 'const' to go on the left to start the declaration, but in C const should go on the right (east const convention)
    switch (emitter->config.dialect) {
        case CDialect_C11: {
            if (as != AsGeneric) warn_print_once(c11_non_generic_as, "warning: standard C does not have address spaces\n");
            prefix = "";
            if (constant)
                name = format_string_arena(emitter->arena->arena, "const %s", name);
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
                    name = format_string_arena(emitter->arena->arena, "__shady_private_globals.%s", name);
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
            name = format_string_arena(emitter->arena->arena, "uniform %s", name);
        else
            name = format_string_arena(emitter->arena->arena, "varying %s", name);
    }

    if (init)
        print(emitter->fn_decls, "\n%s%s = %s;", prefix, c_emit_type(emitter, type, name), init);
    else
        print(emitter->fn_decls, "\n%s%s;", prefix, c_emit_type(emitter, type, name));

    //if (!has_forward_declarations(emitter->config.dialect) || !init)
    //    return;
    //
    //String declaration = c_emit_type(emitter, type, decl_center);
    //print(emitter->fn_decls, "\n%s;", declaration);
}

void c_emit_decl(Emitter* emitter, const Node* decl) {
    assert(is_declaration(decl));

    CTerm* found = lookup_existing_term(emitter, NULL, decl);
    if (found) return;

    CType* found2 = lookup_existing_type(emitter, decl);
    if (found2) return;

    const char* name = c_legalize_identifier(emitter, get_declaration_name(decl));
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    CTerm emit_as;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            String init = NULL;
            if (decl->payload.global_variable.init)
                init = to_cvalue(emitter, c_emit_value(emitter, NULL, decl->payload.global_variable.init));
            AddressSpace ass = decl->payload.global_variable.address_space;
            if (ass == AsInput || ass == AsOutput)
                init = NULL;

            const GlobalVariable* gvar = &decl->payload.global_variable;
            if (is_decl_builtin(decl)) {
                Builtin b = get_decl_builtin(decl);
                CTerm t = c_emit_builtin(emitter, b);
                register_emitted(emitter, NULL, decl, t);
                return;
            }

            if (ass == AsOutput && emitter->compiler_config->specialization.execution_model == EmFragment) {
                int location = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(lookup_annotation(decl, "Location"))), false);
                CTerm t = term_from_cvar(format_string_interned(emitter->arena, "gl_FragData[%d]", location));
                register_emitted(emitter, NULL, decl, t);
                return;
            }

            decl_type = decl->payload.global_variable.type;
            // we emit the global variable as a CVar, so we can refer to it's 'address' without explicit ptrs
            emit_as = term_from_cvar(name);
            if ((decl->payload.global_variable.address_space == AsPrivate) && emitter->config.dialect == CDialect_CUDA) {
                if (emitter->use_private_globals) {
                    register_emitted(emitter, NULL, decl, term_from_cvar(format_string_arena(emitter->arena->arena, "__shady_private_globals->%s", name)));
                    // HACK
                    return;
                }
                emit_as = term_from_cvar(format_string_interned(emitter->arena, "__shady_thread_local_access(%s)", name));
                if (init)
                    init = format_string_interned(emitter->arena, "__shady_replicate_thread_local(%s)", init);
                register_emitted(emitter, NULL, decl, emit_as);
            }
            register_emitted(emitter, NULL, decl, emit_as);

            AddressSpace as = decl->payload.global_variable.address_space;
            c_emit_global_variable_definition(emitter, as, decl_center, decl_type, false, init);
            return;
        }
        case Function_TAG: {
            emit_as = term_from_cvalue(name);
            register_emitted(emitter, NULL, decl, emit_as);
            String head = c_emit_fn_head(emitter, decl->type, name, decl);
            const Node* body = decl->payload.fun.body;
            if (body) {
                FnEmitter fn = {
                    .cfg = build_fn_cfg(decl),
                    .emitted_terms = new_dict(Node*, CTerm, (HashFn) hash_node, (CmpFn) compare_node),
                };
                fn.scheduler = new_scheduler(fn.cfg);
                fn.instruction_printers = calloc(sizeof(Printer*), fn.cfg->size);
                // for (size_t i = 0; i < fn.cfg->size; i++)
                //     fn.instruction_printers[i] = open_growy_as_printer(new_growy());

                for (size_t i = 0; i < decl->payload.fun.params.count; i++) {
                    String param_name;
                    String variable_name = get_value_name_unsafe(decl->payload.fun.params.nodes[i]);
                    param_name = format_string_interned(emitter->arena, "%s_%d", c_legalize_identifier(emitter, variable_name), decl->payload.fun.params.nodes[i]->id);
                    register_emitted(emitter, &fn, decl->payload.fun.params.nodes[i], term_from_cvalue(param_name));
                }

                String fn_body = c_emit_body(emitter, &fn, decl);
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
                print(emitter->fn_defs, "\n%s { ", head);
                indent(emitter->fn_defs);
                print(emitter->fn_defs, " %s", fn_body);
                deindent(emitter->fn_defs);
                print(emitter->fn_defs, "\n}");

                destroy_scheduler(fn.scheduler);
                destroy_cfg(fn.cfg);
                destroy_dict(fn.emitted_terms);
                free(fn.instruction_printers);
            }

            print(emitter->fn_decls, "\n%s;", head);
            return;
        }
        case Constant_TAG: {
            emit_as = term_from_cvalue(name);
            register_emitted(emitter, NULL, decl, emit_as);

            String init = to_cvalue(emitter, c_emit_value(emitter, NULL, decl->payload.constant.value));
            c_emit_global_variable_definition(emitter, AsGlobal, decl_center, decl->type, true, init);
            return;
        }
        case NominalType_TAG: {
            CType emitted = name;
            register_emitted_type(emitter, decl, emitted);
            switch (emitter->config.dialect) {
                case CDialect_ISPC:
                default: print(emitter->type_decls, "\ntypedef %s;", c_emit_type(emitter, decl->payload.nom_type.body, emitted)); break;
                case CDialect_GLSL: c_emit_nominal_type_body(emitter, format_string_arena(emitter->arena->arena, "struct %s /* nominal */", emitted), decl->payload.nom_type.body); break;
            }
            return;
        }
        default: error("not a decl");
    }
}

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
        print(emitter.type_decls, "\ntypedef %s;\n", c_emit_type(&emitter, arr_type(arena, (ArrType) {
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
        c_emit_decl(&emitter, decls.nodes[i]);

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
