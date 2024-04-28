#include "shady/ir.h"

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

        .target = default_target_config(),

        .specialization = {
            .subgroup_size = 8,
            .entry_point = NULL
        }
    };
}

TargetConfig default_target_config() {
    return (TargetConfig) {
        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
        },
    };
}

ArenaConfig default_arena_config(const TargetConfig* target) {
    ArenaConfig config = {
        .is_simt = true,
        .name_bound = true,
        .allow_fold = true,
        .check_types = true,
        .validate_builtin_types = true,
        .check_op_classes = true,

        .optimisations = {
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