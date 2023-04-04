#include "shady/cli.h"
#include "compile.h"

#include "parser/parser.h"
#include "shady_scheduler_src.h"
#include "transform/internal_constants.h"
#include "portability.h"
#include "ir_private.h"
#include "util.h"

#include <stdbool.h>

#ifdef C_PARSER_PRESENT
#include "../clang-ast/clang_ast.h"
#endif

#ifdef SPV_PARSER_PRESENT
#include "../spirv/s2s.h"
#endif

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .allow_frontend_syntax = false,
        .dynamic_scheduling = true,
        .per_thread_stack_size = 1 KiB,
        .per_subgroup_stack_size = 1 KiB,

        .subgroup_size = 32,

        .target_spirv_version = {
            .major = 1,
            .minor = 4
        },

        .logging = {
            // most of the time, we are not interested in seeing generated/builtin code in the debug output
            .skip_builtin = true,
            .skip_generated = true,
        }
    };
}

ArenaConfig default_arena_config() {
    return (ArenaConfig) {
        .is_simt = true,

        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
        }
    };
}

#define mod (*pmod)

CompilationResult run_compiler_passes(CompilerConfig* config, Module** pmod) {
    ArenaConfig aconfig = get_module_arena(mod)->config;
    Module* old_mod;

    IrArena* old_arena = NULL;
    IrArena* tmp_arena = NULL;

    generate_dummy_constants(config, mod);

    aconfig.name_bound = true;
    RUN_PASS(bind_program)
    RUN_PASS(normalize)

    // TODO: do this late
    patch_constants(config, mod);

    RUN_PASS(reconvergence_heuristics)

    aconfig.check_types = true;
    RUN_PASS(infer_program)

    aconfig.allow_fold = true;

    RUN_PASS(setup_stack_frames)
    RUN_PASS(mark_leaf_functions)

    RUN_PASS(lower_cf_instrs)
    RUN_PASS(opt_restructurize)

    RUN_PASS(lower_callf)
    RUN_PASS(opt_simplify_cf)

    RUN_PASS(lower_continuations)

    RUN_PASS(opt_simplify_cf)
    RUN_PASS(opt_stack)

    RUN_PASS(lower_tailcalls)

    RUN_PASS(eliminate_constants)

    aconfig.subgroup_mask_representation = SubgroupMaskInt64;
    RUN_PASS(lower_mask)
    RUN_PASS(lower_memcpy)
    RUN_PASS(lower_subgroup_ops)
    RUN_PASS(lower_stack)

    RUN_PASS(lower_lea)
    RUN_PASS(lower_generic_ptrs)
    RUN_PASS(lower_physical_ptrs)
    RUN_PASS(lower_subgroup_vars)
    RUN_PASS(lower_memory_layout)

    RUN_PASS(lower_int)

    if (config->lower.simt_to_explicit_simd) {
        aconfig.is_simt = false;
        RUN_PASS(simt2d)
    }

    return CompilationNoError;
}

#undef mod

CompilationResult parse_files(CompilerConfig* config, size_t num_files, const char** file_names, const char** files_contents, Module* mod) {
    ParserConfig pconfig = {
        .front_end = config->allow_frontend_syntax
    };

    for (size_t i = 0; i < num_files; i++) {
        if (file_names && string_ends_with(file_names[i], ".c")) {
#ifdef C_PARSER_PRESENT
            parse_c_file(file_names[i], mod);
#else
            assert(false && "C front-end missing in this version");
#endif
        } else if (file_names && string_ends_with(file_names[i], ".spv")) {
#ifdef SPV_PARSER_PRESENT
            size_t size;
            unsigned char* data;
            bool ok = read_file(file_names[i], &size, &data);
            assert(ok);
            parse_spirv_into_shady(mod, size, (uint32_t*) data);
#else
            assert(false && "SPIR-V front-end missing in this version");
#endif
        } else {
            const char* file_contents;

            if (files_contents) {
                file_contents = files_contents[i];
            } else {
                assert(file_names);
                bool ok = read_file(file_names[i], NULL, &file_contents);
                assert(ok);

                if (file_contents == NULL) {
                    error_print("file does not exist\n");
                    exit(InputFileDoesNotExist);
                }
            }

            debugv_print("Parsing: \n%s\n", file_contents);

            parse(pconfig, file_contents, mod);

            if (!files_contents)
                free((void*) file_contents);
        }
    }

    if (config->dynamic_scheduling) {
        debugv_print("Parsing builtin scheduler code");
        parse(pconfig, shady_scheduler_src, mod);
    }

    // Free the read files
    for (size_t i = 0; i < num_files; i++)

    return CompilationNoError;
}
