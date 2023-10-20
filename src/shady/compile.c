#include "shady/driver.h"
#include "compile.h"

#include "parser/parser.h"
#include "shady_scheduler_src.h"
#include "transform/internal_constants.h"
#include "portability.h"
#include "ir_private.h"
#include "util.h"

#include <stdbool.h>

#ifdef LLVM_PARSER_PRESENT
#include "../frontends/llvm/l2s.h"
#endif

#ifdef SPV_PARSER_PRESENT
#include "../frontends/spirv/s2s.h"
#endif

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .dynamic_scheduling = true,
        .per_thread_stack_size = 4 KiB,
        .per_subgroup_stack_size = 1 KiB,

        .target_spirv_version = {
            .major = 1,
            .minor = 4
        },

        .logging = {
            // most of the time, we are not interested in seeing generated & internal code in the debug output
            .skip_internal = true,
            .skip_generated = true,
        },

        .specialization = {
            .subgroup_size = 8,
            .entry_point = NULL
        }
    };
}

ArenaConfig default_arena_config() {
    return (ArenaConfig) {
        .is_simt = true,
        .validate_builtin_types = false,

        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
        }
    };
}

#define mod (*pmod)

CompilationResult run_compiler_passes(CompilerConfig* config, Module** pmod) {
    if (config->dynamic_scheduling) {
        debugv_print("Parsing builtin scheduler code");
        ParserConfig pconfig = {
            .front_end = true,
        };
        parse_shady_ir(pconfig, shady_scheduler_src, mod);
    }

    ArenaConfig aconfig = get_module_arena(mod)->config;

    aconfig.specializations.subgroup_size = config->specialization.subgroup_size;

    Module* old_mod;

    IrArena* old_arena = NULL;
    IrArena* tmp_arena = NULL;

    generate_dummy_constants(config, mod);

    aconfig.name_bound = true;
    RUN_PASS(bind_program)
    RUN_PASS(normalize)

    aconfig.check_types = true;
    RUN_PASS(infer_program)

    aconfig.validate_builtin_types = true;
    RUN_PASS(normalize_builtins);

    aconfig.allow_fold = true;
    RUN_PASS(opt_inline_jumps)

    RUN_PASS(lcssa)
    RUN_PASS(reconvergence_heuristics)

    RUN_PASS(setup_stack_frames)
    RUN_PASS(lower_cf_instrs)
    if (!config->hacks.force_join_point_lifting) {
        RUN_PASS(mark_leaf_functions)
    }

    RUN_PASS(lower_callf)
    RUN_PASS(opt_inline)

    RUN_PASS(lift_indirect_targets)

    RUN_PASS(opt_stack)

    RUN_PASS(lower_tailcalls)
    RUN_PASS(lower_switch_btree)
    RUN_PASS(opt_restructurize)
    RUN_PASS(opt_inline_jumps)

    aconfig.specializations.subgroup_mask_representation = SubgroupMaskInt64;
    RUN_PASS(lower_mask)
    RUN_PASS(lower_memcpy)
    RUN_PASS(lower_subgroup_ops)
    RUN_PASS(lower_stack)

    RUN_PASS(lower_lea)
    RUN_PASS(lower_generic_ptrs)
    RUN_PASS(lower_physical_ptrs)
    RUN_PASS(lower_subgroup_vars)
    RUN_PASS(lower_memory_layout)

    if (config->lower.decay_ptrs) {
        RUN_PASS(lower_decay_ptrs)
    }

    RUN_PASS(lower_int)

    if (config->lower.simt_to_explicit_simd) {
        aconfig.is_simt = false;
        RUN_PASS(simt2d)
    }

    if (config->specialization.entry_point) {
        specialize_arena_config(&aconfig, mod, config);
        RUN_PASS(specialize_for_entry_point)
    }
    RUN_PASS(lower_fill)

    return CompilationNoError;
}

#undef mod
