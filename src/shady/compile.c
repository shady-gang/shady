#include "shady/driver.h"
#include "compile.h"

#include "parser/parser.h"
#include "shady_scheduler_src.h"
#include "transform/internal_constants.h"
#include "portability.h"
#include "ir_private.h"
#include "util.h"

#include <stdbool.h>

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .dynamic_scheduling = true,
        .per_thread_stack_size = 4 KiB,

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
        .allow_subgroup_memory = true,
        .allow_shared_memory = true,

        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
        },
    };
}

CompilationResult run_compiler_passes(CompilerConfig* config, Module** pmod) {
    if (config->dynamic_scheduling) {
        debugv_print("Parsing builtin scheduler code");
        ParserConfig pconfig = {
            .front_end = true,
        };
        parse_shady_ir(pconfig, shady_scheduler_src, *pmod);
    }

    IrArena* initial_arena = (*pmod)->arena;
    Module* old_mod = NULL;

    generate_dummy_constants(config, *pmod);

    if (!get_module_arena(*pmod)->config.name_bound)
        RUN_PASS(bind_program)
    RUN_PASS(normalize)

    RUN_PASS(infer_program)
    RUN_PASS(normalize_builtins);

    RUN_PASS(opt_inline_jumps)

    RUN_PASS(lcssa)
    RUN_PASS(reconvergence_heuristics)

    RUN_PASS(lower_cf_instrs)
    RUN_PASS(opt_mem2reg)
    RUN_PASS(setup_stack_frames)
    if (!config->hacks.force_join_point_lifting)
        RUN_PASS(mark_leaf_functions)

    RUN_PASS(lower_callf)
    RUN_PASS(opt_inline)

    RUN_PASS(lift_indirect_targets)

    if (config->specialization.execution_model != EmNone)
        RUN_PASS(specialize_execution_model)

    RUN_PASS(opt_stack)

    RUN_PASS(lower_tailcalls)
    RUN_PASS(lower_switch_btree)
    RUN_PASS(opt_restructurize)
    RUN_PASS(opt_inline_jumps)

    RUN_PASS(lower_mask)
    RUN_PASS(lower_memcpy)
    RUN_PASS(lower_subgroup_ops)
    RUN_PASS(lower_stack)

    RUN_PASS(lower_lea)
    RUN_PASS(lower_generic_ptrs)
    RUN_PASS(lower_physical_ptrs)
    RUN_PASS(lower_subgroup_vars)
    RUN_PASS(lower_memory_layout)

    if (config->lower.decay_ptrs)
        RUN_PASS(lower_decay_ptrs)

    RUN_PASS(lower_int)

    if (config->lower.simt_to_explicit_simd)
        RUN_PASS(simt2d)

    if (config->specialization.entry_point)
        RUN_PASS(specialize_entry_point)
    RUN_PASS(lower_fill)

    return CompilationNoError;
}

#undef mod
