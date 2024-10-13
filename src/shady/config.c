#include "shady/ir.h"
#include "shady/config.h"

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig shd_default_compiler_config(void) {
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

        /*.shader_diagnostics = {
            .max_top_iterations = 10,
        },

        .printf_trace = {
            .god_function = true,
        },*/

        .target = shd_default_target_config(),

        .specialization = {
            .subgroup_size = 8,
            .entry_point = NULL
        }
    };
}

TargetConfig shd_default_target_config(void) {
    return (TargetConfig) {
        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
        },
    };
}

ArenaConfig shd_default_arena_config(const TargetConfig* target) {
    ArenaConfig config = {
        .is_simt = true,
        .name_bound = true,
        .allow_fold = true,
        .check_types = true,
        .validate_builtin_types = true,
        .check_op_classes = true,

        .optimisations = {
            .inline_single_use_bbs = true,
            .fold_static_control_flow = true,
            .delete_unreachable_structured_cases = true,
        },

        .memory = target->memory
    };

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        // by default, all address spaces are physical !
        config.address_spaces[i].physical = true;
        config.address_spaces[i].allowed = true;
    }

    return config;
}
