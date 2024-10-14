#include "emit_c.h"

#include "../shady/ir_private.h"
#include "../shady/passes/passes.h"
#include "../shady/analysis/cfg.h"
#include "../shady/analysis/scheduler.h"

#include "shady_cuda_prelude_src.h"
#include "shady_cuda_runtime_src.h"
#include "shady_glsl_runtime_120_src.h"
#include "shady_ispc_runtime_src.h"

#include "portability.h"
#include "dict.h"
#include "log.h"
#include "util.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

void shd_c_register_emitted(Emitter* emitter, FnEmitter* fn, const Node* node, CTerm as) {
    //assert(as.value || as.var);
    shd_dict_insert(const Node*, CTerm, fn ? fn->emitted_terms : emitter->emitted_terms, node, as);
}

CTerm* shd_c_lookup_existing_term(Emitter* emitter, FnEmitter* fn, const Node* node) {
    CTerm* found = NULL;
    if (fn)
        found = shd_dict_find_value(const Node*, CTerm, fn->emitted_terms, node);
    if (!found)
        found = shd_dict_find_value(const Node*, CTerm, emitter->emitted_terms, node);
    return found;
}

void shd_c_register_emitted_type(Emitter* emitter, const Node* node, String as) {
    shd_dict_insert(const Node*, String, emitter->emitted_types, node, as);
}

CType* shd_c_lookup_existing_type(Emitter* emitter, const Type* node) {
    CType* found = shd_dict_find_value(const Node*, CType, emitter->emitted_types, node);
    return found;
}

CValue shd_c_to_ssa(SHADY_UNUSED Emitter* e, CTerm term) {
    if (term.value)
        return term.value;
    if (term.var)
        return shd_format_string_arena(e->arena->arena, "(&%s)", term.var);
    assert(false);
}

CAddr shd_c_deref(Emitter* e, CTerm term) {
    if (term.value)
        return shd_format_string_arena(e->arena->arena, "(*%s)", term.value);
    if (term.var)
        return term.var;
    assert(false);
}

// TODO: utf8
static bool is_legal_identifier_char(char c) {
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

String shd_c_legalize_identifier(Emitter* e, String src) {
    if (!src)
        return "unnamed";
    size_t len = strlen(src);
    LARRAY(char, dst, len + 1);
    size_t i;
    for (i = 0; i < len; i++) {
        char c = src[i];
        if (is_legal_identifier_char(c))
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
CTerm shd_ispc_varying_ptr_helper(Emitter* emitter, Printer* block_printer, const Type* ptr_type, CTerm term) {
    String interm = shd_make_unique_name(emitter->arena, "intermediary_ptr_value");
    assert(ptr_type->tag == PtrType_TAG);
    const Type* ut = shd_as_qualified_type(ptr_type, true);
    const Type* vt = shd_as_qualified_type(ptr_type, false);
    String lhs = shd_c_emit_type(emitter, vt, interm);
    shd_print(block_printer, "\n%s = ((%s) %s) + programIndex;", lhs, shd_c_emit_type(emitter, ut, NULL), shd_c_to_ssa(emitter, term));
    return term_from_cvalue(interm);
}

void shd_c_emit_variable_declaration(Emitter* emitter, Printer* block_printer, const Type* t, String variable_name, bool mut, const CTerm* initializer) {
    assert((mut || initializer != NULL) && "unbound results are only allowed when creating a mutable local variable");

    String prefix = "";
    String center = variable_name;

    // add extra qualifiers if immutable
    if (!mut) switch (emitter->config.dialect) {
        case CDialect_ISPC:
            center = shd_format_string_arena(emitter->arena->arena, "const %s", center);
            break;
        case CDialect_C11:
        case CDialect_CUDA:
            center = shd_format_string_arena(emitter->arena->arena, "const %s", center);
            break;
        case CDialect_GLSL:
            if (emitter->config.glsl_version >= 130)
                prefix = "const ";
            break;
    }

    String decl = shd_c_emit_type(emitter, t, center);
    if (initializer)
        shd_print(block_printer, "\n%s%s = %s;", prefix, decl, shd_c_to_ssa(emitter, *initializer));
    else
        shd_print(block_printer, "\n%s%s;", prefix, decl);
}

void shd_c_emit_pack_code(Printer* p, Strings src, String dst) {
    for (size_t i = 0; i < src.count; i++) {
        shd_print(p, "\n%s->_%d = %s", dst, src.strings[i], i);
    }
}

void shd_c_emit_unpack_code(Printer* p, String src, Strings dst) {
    for (size_t i = 0; i < dst.count; i++) {
        shd_print(p, "\n%s = %s->_%d", dst.strings[i], src, i);
    }
}

void shd_c_emit_global_variable_definition(Emitter* emitter, AddressSpace as, String name, const Type* type, bool constant, String init) {
    String prefix = NULL;

    bool is_fs = emitter->compiler_config->specialization.execution_model == EmFragment;
    // GLSL wants 'const' to go on the left to start the declaration, but in C const should go on the right (east const convention)
    switch (emitter->config.dialect) {
        case CDialect_C11: {
            if (as != AsGeneric) shd_warn_print_once(c11_non_generic_as, "warning: standard C does not have address spaces\n");
            prefix = "";
            if (constant)
                name = shd_format_string_arena(emitter->arena->arena, "const %s", name);
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
                    name = shd_format_string_arena(emitter->arena->arena, "__shady_private_globals.%s", name);
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
                    prefix = shd_format_string_arena(emitter->arena->arena, "/* %s */", shd_get_address_space_name(as));
                    shd_warn_print("warning: address space %s not supported in CUDA for global variables\n", shd_get_address_space_name(as));
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
                    prefix = shd_format_string_arena(emitter->arena->arena, "/* %s */", shd_get_address_space_name(as));
                    shd_warn_print("warning: address space %s not supported in GLSL for global variables\n", shd_get_address_space_name(as));
                    break;
                }
            }
            break;
    }

    assert(prefix);

    // ISPC wants uniform/varying annotations
    if (emitter->config.dialect == CDialect_ISPC) {
        bool uniform = shd_is_addr_space_uniform(emitter->arena, as);
        if (uniform)
            name = shd_format_string_arena(emitter->arena->arena, "uniform %s", name);
        else
            name = shd_format_string_arena(emitter->arena->arena, "varying %s", name);
    }

    if (init)
        shd_print(emitter->fn_decls, "\n%s%s = %s;", prefix, shd_c_emit_type(emitter, type, name), init);
    else
        shd_print(emitter->fn_decls, "\n%s%s;", prefix, shd_c_emit_type(emitter, type, name));

    //if (!has_forward_declarations(emitter->config.dialect) || !init)
    //    return;
    //
    //String declaration = c_emit_type(emitter, type, decl_center);
    //shd_print(emitter->fn_decls, "\n%s;", declaration);
}

void shd_c_emit_decl(Emitter* emitter, const Node* decl) {
    assert(is_declaration(decl));

    CTerm* found = shd_c_lookup_existing_term(emitter, NULL, decl);
    if (found) return;

    CType* found2 = shd_c_lookup_existing_type(emitter, decl);
    if (found2) return;

    const char* name = shd_c_legalize_identifier(emitter, get_declaration_name(decl));
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    CTerm emit_as;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            String init = NULL;
            if (decl->payload.global_variable.init)
                init = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, NULL, decl->payload.global_variable.init));
            AddressSpace ass = decl->payload.global_variable.address_space;
            if (ass == AsInput || ass == AsOutput)
                init = NULL;

            const GlobalVariable* gvar = &decl->payload.global_variable;
            if (shd_is_decl_builtin(decl)) {
                Builtin b = shd_get_decl_builtin(decl);
                CTerm t = shd_c_emit_builtin(emitter, b);
                shd_c_register_emitted(emitter, NULL, decl, t);
                return;
            }

            if (ass == AsOutput && emitter->compiler_config->specialization.execution_model == EmFragment) {
                int location = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(shd_lookup_annotation(decl, "Location"))), false);
                CTerm t = term_from_cvar(shd_fmt_string_irarena(emitter->arena, "gl_FragData[%d]", location));
                shd_c_register_emitted(emitter, NULL, decl, t);
                return;
            }

            decl_type = decl->payload.global_variable.type;
            // we emit the global variable as a CVar, so we can refer to it's 'address' without explicit ptrs
            emit_as = term_from_cvar(name);
            if ((decl->payload.global_variable.address_space == AsPrivate) && emitter->config.dialect == CDialect_CUDA) {
                if (emitter->use_private_globals) {
                    shd_c_register_emitted(emitter, NULL, decl, term_from_cvar(shd_format_string_arena(emitter->arena->arena, "__shady_private_globals->%s", name)));
                    // HACK
                    return;
                }
                emit_as = term_from_cvar(shd_fmt_string_irarena(emitter->arena, "__shady_thread_local_access(%s)", name));
                if (init)
                    init = shd_fmt_string_irarena(emitter->arena, "__shady_replicate_thread_local(%s)", init);
                shd_c_register_emitted(emitter, NULL, decl, emit_as);
            }
            shd_c_register_emitted(emitter, NULL, decl, emit_as);

            AddressSpace as = decl->payload.global_variable.address_space;
            shd_c_emit_global_variable_definition(emitter, as, decl_center, decl_type, false, init);
            return;
        }
        case Function_TAG: {
            emit_as = term_from_cvalue(name);
            shd_c_register_emitted(emitter, NULL, decl, emit_as);
            String head = shd_c_emit_fn_head(emitter, decl->type, name, decl);
            const Node* body = decl->payload.fun.body;
            if (body) {
                FnEmitter fn = {
                    .cfg = build_fn_cfg(decl),
                    .emitted_terms = shd_new_dict(Node*, CTerm, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
                };
                fn.scheduler = shd_new_scheduler(fn.cfg);
                fn.instruction_printers = calloc(sizeof(Printer*), fn.cfg->size);
                // for (size_t i = 0; i < fn.cfg->size; i++)
                //     fn.instruction_printers[i] = open_growy_as_printer(new_growy());

                for (size_t i = 0; i < decl->payload.fun.params.count; i++) {
                    String param_name;
                    String variable_name = shd_get_value_name_unsafe(decl->payload.fun.params.nodes[i]);
                    param_name = shd_fmt_string_irarena(emitter->arena, "%s_%d", shd_c_legalize_identifier(emitter, variable_name), decl->payload.fun.params.nodes[i]->id);
                    shd_c_register_emitted(emitter, &fn, decl->payload.fun.params.nodes[i], term_from_cvalue(param_name));
                }

                String fn_body = shd_c_emit_body(emitter, &fn, decl);
                if (emitter->config.dialect == CDialect_ISPC) {
                    // ISPC hack: This compiler (like seemingly all LLVM-based compilers) has broken handling of the execution mask - it fails to generated masked stores for the entry BB of a function that may be called non-uniformingly
                    // therefore we must tell ISPC to please, pretty please, mask everything by branching on what the mask should be
                    fn_body = shd_format_string_arena(emitter->arena->arena, "if ((lanemask() >> programIndex) & 1u) { %s}", fn_body);
                    // I hate everything about this too.
                } else if (emitter->config.dialect == CDialect_CUDA) {
                    if (shd_lookup_annotation(decl, "EntryPoint")) {
                        // fn_body = format_string_arena(emitter->arena->arena, "\n__shady_entry_point_init();%s", fn_body);
                        if (emitter->use_private_globals) {
                            fn_body = shd_format_string_arena(emitter->arena->arena, "\n__shady_PrivateGlobals __shady_private_globals_alloc;\n __shady_PrivateGlobals* __shady_private_globals = &__shady_private_globals_alloc;\n%s", fn_body);
                        }
                        fn_body = shd_format_string_arena(emitter->arena->arena, "\n__shady_prepare_builtins();%s", fn_body);
                    }
                }
                shd_print(emitter->fn_defs, "\n%s { ", head);
                shd_printer_indent(emitter->fn_defs);
                shd_print(emitter->fn_defs, " %s", fn_body);
                shd_printer_deindent(emitter->fn_defs);
                shd_print(emitter->fn_defs, "\n}");

                shd_destroy_scheduler(fn.scheduler);
                shd_destroy_cfg(fn.cfg);
                shd_destroy_dict(fn.emitted_terms);
                free(fn.instruction_printers);
            }

            shd_print(emitter->fn_decls, "\n%s;", head);
            return;
        }
        case Constant_TAG: {
            emit_as = term_from_cvalue(name);
            shd_c_register_emitted(emitter, NULL, decl, emit_as);

            String init = shd_c_to_ssa(emitter, shd_c_emit_value(emitter, NULL, decl->payload.constant.value));
            shd_c_emit_global_variable_definition(emitter, AsGlobal, decl_center, decl->type, true, init);
            return;
        }
        case NominalType_TAG: {
            CType emitted = name;
            shd_c_register_emitted_type(emitter, decl, emitted);
            switch (emitter->config.dialect) {
                case CDialect_ISPC:
                default: shd_print(emitter->type_decls, "\ntypedef %s;", shd_c_emit_type(emitter, decl->payload.nom_type.body, emitted)); break;
                case CDialect_GLSL: shd_c_emit_nominal_type_body(emitter, shd_format_string_arena(emitter->arena->arena, "struct %s /* nominal */", emitted), decl->payload.nom_type.body); break;
            }
            return;
        }
        default: shd_error("not a decl");
    }
}

static Module* run_backend_specific_passes(const CompilerConfig* config, CEmitterConfig* econfig, Module* initial_mod) {
    IrArena* initial_arena = initial_mod->arena;
    Module** pmod = &initial_mod;

    // C lacks a nice way to express constants that can be used in type definitions afterwards, so let's just inline them all.
    RUN_PASS(shd_pass_eliminate_constants)
    if (econfig->dialect == CDialect_ISPC) {
        RUN_PASS(shd_pass_lower_workgroups)
    }
    if (econfig->dialect != CDialect_GLSL) {
        RUN_PASS(shd_pass_lower_vec_arr)
    }
    return *pmod;
}

static String collect_private_globals_in_struct(Emitter* emitter, Module* m) {
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);

    shd_print(p, "typedef struct __shady_PrivateGlobals {\n");
    Nodes decls = shd_module_get_declarations(m);
    size_t count = 0;
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG)
            continue;
        AddressSpace as = decl->payload.global_variable.address_space;
        if (as != AsPrivate)
            continue;
        shd_print(p, "%s;\n", shd_c_emit_type(emitter, decl->payload.global_variable.type, decl->payload.global_variable.name));
        count++;
    }
    shd_print(p, "} __shady_PrivateGlobals;\n");

    if (count == 0) {
        shd_destroy_printer(p);
        shd_destroy_growy(g);
        return NULL;
    }
    return shd_printer_growy_unwrap(p);
}

CEmitterConfig shd_default_c_emitter_config(void) {
    return (CEmitterConfig) {
        .glsl_version = 420,
    };
}

void shd_emit_c(const CompilerConfig* compiler_config, CEmitterConfig config, Module* mod, size_t* output_size, char** output, Module** new_mod) {
    IrArena* initial_arena = shd_module_get_arena(mod);
    mod = run_backend_specific_passes(compiler_config, &config, mod);
    IrArena* arena = shd_module_get_arena(mod);

    Growy* type_decls_g = shd_new_growy();
    Growy* fn_decls_g = shd_new_growy();
    Growy* fn_defs_g = shd_new_growy();

    Emitter emitter = {
        .compiler_config = compiler_config,
        .config = config,
        .arena = arena,
        .type_decls = shd_new_printer_from_growy(type_decls_g),
        .fn_decls = shd_new_printer_from_growy(fn_decls_g),
        .fn_defs = shd_new_printer_from_growy(fn_defs_g),
        .emitted_terms = shd_new_dict(Node*, CTerm, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .emitted_types = shd_new_dict(Node*, String, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };

    Growy* final = shd_new_growy();
    Printer* finalp = shd_new_printer_from_growy(final);

    shd_print(finalp, "/* file generated by shady */\n");

    switch (emitter.config.dialect) {
        case CDialect_ISPC: {
            shd_print(emitter.fn_defs, shady_ispc_runtime_src);
            break;
        }
        case CDialect_C11:
            shd_print(finalp, "\n#include <stdbool.h>");
            shd_print(finalp, "\n#include <stdint.h>");
            shd_print(finalp, "\n#include <stddef.h>");
            shd_print(finalp, "\n#include <stdio.h>");
            shd_print(finalp, "\n#include <math.h>");
            break;
        case CDialect_GLSL:
            shd_print(finalp, "#version %d\n", emitter.config.glsl_version);
            if (emitter.need_64b_ext)
                shd_print(finalp, "#extension GL_ARB_gpu_shader_int64: require\n");
            shd_print(finalp, "#define ubyte uint\n");
            shd_print(finalp, "#define uchar uint\n");
            shd_print(finalp, "#define ulong uint\n");
            if (emitter.config.glsl_version <= 120)
                shd_print(finalp, shady_glsl_runtime_120_src);
            break;
        case CDialect_CUDA: {
            size_t total_workgroup_size = emitter.arena->config.specializations.workgroup_size[0];
            total_workgroup_size *= emitter.arena->config.specializations.workgroup_size[1];
            total_workgroup_size *= emitter.arena->config.specializations.workgroup_size[2];

            shd_print(finalp, "#define __shady_workgroup_size %d\n", total_workgroup_size);
            shd_print(finalp, "#define __shady_replicate_thread_local(v) { ");
            for (size_t i = 0; i < total_workgroup_size; i++)
                shd_print(finalp, "v, ");
            shd_print(finalp, "}\n");
            shd_print(finalp, shady_cuda_prelude_src);

            shd_print(emitter.type_decls, "\ntypedef %s;\n", shd_c_emit_type(&emitter, arr_type(arena, (ArrType) {
                .size = shd_int32_literal(arena, 3),
                .element_type = shd_uint32_type(arena)
            }), "uvec3"));
            shd_print(emitter.fn_defs, shady_cuda_runtime_src);

            String private_globals = collect_private_globals_in_struct(&emitter, mod);
            if (private_globals) {
                emitter.use_private_globals = true;
                shd_print(emitter.type_decls, private_globals);
                free((void*) private_globals);
            }
            break;
        }
        default: break;
    }

    Nodes decls = shd_module_get_declarations(mod);
    for (size_t i = 0; i < decls.count; i++)
        shd_c_emit_decl(&emitter, decls.nodes[i]);

    shd_print(finalp, "\n/* types: */\n");
    shd_growy_append_bytes(final, shd_growy_size(type_decls_g), shd_growy_data(type_decls_g));

    shd_print(finalp, "\n/* declarations: */\n");
    shd_growy_append_bytes(final, shd_growy_size(fn_decls_g), shd_growy_data(fn_decls_g));

    shd_print(finalp, "\n/* definitions: */\n");
    shd_growy_append_bytes(final, shd_growy_size(fn_defs_g), shd_growy_data(fn_defs_g));

    shd_print(finalp, "\n");
    shd_print(finalp, "\n");
    shd_print(finalp, "\n");
    shd_growy_append_bytes(final, 1, "\0");

    shd_destroy_printer(emitter.type_decls);
    shd_destroy_printer(emitter.fn_decls);
    shd_destroy_printer(emitter.fn_defs);

    shd_destroy_growy(type_decls_g);
    shd_destroy_growy(fn_decls_g);
    shd_destroy_growy(fn_defs_g);

    shd_destroy_dict(emitter.emitted_types);
    shd_destroy_dict(emitter.emitted_terms);

    *output_size = shd_growy_size(final) - 1;
    *output = shd_growy_deconstruct(final);
    shd_destroy_printer(finalp);

    if (new_mod)
        *new_mod = mod;
    else if (initial_arena != arena)
        shd_destroy_ir_arena(arena);
}
