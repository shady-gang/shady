#include "shady/driver.h"
#include "compile.h"

#include "frontends/slim/parser.h"
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

        .lower = {
            .emulate_physical_memory = true,
            .emulate_generic_ptrs = true,
        },

        .logging = {
            // most of the time, we are not interested in seeing generated & internal code in the debug output
            //.print_internal = true,
            //.print_generated = true,
            .print_builtin = true,
        },

        .optimisations = {
            .cleanup = {
                .after_every_pass = true,
                .delete_unused_instructions = true,
            }
        },

        .specialization = {
            .subgroup_size = 8,
            .entry_point = NULL
        }
    };
}

ArenaConfig default_arena_config() {
    ArenaConfig config = {
        .is_simt = true,
        .name_bound = true,
        .allow_fold = true,
        .check_types = true,
        .validate_builtin_types = true,
        .check_op_classes = true,

        .memory = {
            .word_size = IntTy8,
            .ptr_size = IntTy64,
        },

        .optimisations = {
            .delete_unreachable_structured_cases = true,
        },
    };

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        // by default, all address spaces are physical !
        config.address_spaces[i].physical = true;
        config.address_spaces[i].allowed = true;
    }

    return config;
}

void add_scheduler_source(const CompilerConfig* config, Module* dst) {
    ParserConfig pconfig = {
        .front_end = true,
    };
    Module* builtin_scheduler_mod = parse_slim_module(config, pconfig, shady_scheduler_src, "builtin_scheduler");
    debug_print("Adding builtin scheduler code");
    link_module(dst, builtin_scheduler_mod);
    destroy_ir_arena(get_module_arena(builtin_scheduler_mod));
}

CompilationResult run_compiler_passes(CompilerConfig* config, Module** pmod) {
    if (config->dynamic_scheduling) {
        add_scheduler_source(config, *pmod);
    }

    IrArena* initial_arena = (*pmod)->arena;
    Module* old_mod = NULL;

    RUN_PASS(reconvergence_heuristics)

    RUN_PASS(lower_cf_instrs)
    RUN_PASS(opt_mem2reg) // run because control-flow is now normalized
    RUN_PASS(setup_stack_frames)
    if (!config->hacks.force_join_point_lifting)
        RUN_PASS(mark_leaf_functions)

    RUN_PASS(lower_callf)
    RUN_PASS(opt_inline)

    RUN_PASS(lift_indirect_targets)
    RUN_PASS(opt_mem2reg) // run because we can now weaken non-leaking allocas

    if (config->specialization.execution_model != EmNone)
        RUN_PASS(specialize_execution_model)

    RUN_PASS(opt_stack)

    RUN_PASS(lower_tailcalls)
    RUN_PASS(lower_switch_btree)
    RUN_PASS(opt_restructurize)
    RUN_PASS(opt_mem2reg)

    if (config->specialization.entry_point)
        RUN_PASS(specialize_entry_point)

    RUN_PASS(lower_mask)
    RUN_PASS(lower_memcpy)
    RUN_PASS(lower_subgroup_ops)
    if (config->lower.emulate_physical_memory) {
        RUN_PASS(lower_alloca)
    }
    RUN_PASS(lower_stack)
    RUN_PASS(lower_lea)
    RUN_PASS(lower_generic_globals)
    if (config->lower.emulate_generic_ptrs) {
        RUN_PASS(lower_generic_ptrs)
    }
    if (config->lower.emulate_physical_memory) {
        RUN_PASS(lower_physical_ptrs)
    }
    RUN_PASS(lower_subgroup_vars)
    RUN_PASS(lower_memory_layout)

    if (config->lower.decay_ptrs)
        RUN_PASS(lower_decay_ptrs)

    RUN_PASS(lower_int)

    if (config->lower.simt_to_explicit_simd)
        RUN_PASS(simt2d)
    RUN_PASS(lower_fill)
    RUN_PASS(normalize_builtins)

    return CompilationNoError;
}

#undef mod
