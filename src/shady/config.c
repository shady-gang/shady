#include "shady/ir.h"
#include "shady/config.h"

#include "util.h"

#include <stdlib.h>

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig shd_default_compiler_config(void) {
    CompilerConfig config = {
        .dynamic_scheduling = true,
        .per_thread_stack_size = 4 KiB,

        .lower = {
            .emulate_physical_memory = true,
            .emulate_generic_ptrs = true,

            //.use_scratch_for_private = true,
        },

        .optimisations = {
            .cleanup = {
                .after_every_pass = true,
                .delete_unused_instructions = true,
            }
        },

        .target = shd_default_target_config(),
    };

    String trace_opts = getenv("SHADY_PRINTF_TRACE");
    if (trace_opts) {
        shd_configure_bool_flag_in_list(trace_opts, "stack-size", &config.printf_trace.stack_size);
        shd_configure_bool_flag_in_list(trace_opts, "stack-access", &config.printf_trace.stack_accesses);
        shd_configure_bool_flag_in_list(trace_opts, "max-stack-size", &config.printf_trace.max_stack_size);
        shd_configure_bool_flag_in_list(trace_opts, "memory-access", &config.printf_trace.memory_accesses);
        shd_configure_bool_flag_in_list(trace_opts, "top-function", &config.printf_trace.top_function);
        shd_configure_bool_flag_in_list(trace_opts, "subgroup-ops", &config.printf_trace.subgroup_ops);
        shd_configure_bool_flag_in_list(trace_opts, "scratch-base-addr", &config.printf_trace.scratch_base_addr);
    }

    return config;
}

TargetConfig shd_default_target_config(void) {
    TargetConfig config = {
        .memory = {
            .word_size = IntTy32,
            .ptr_size = IntTy64,
            .fn_ptr_size = IntTy32,
            .exec_mask_size = IntTy64
        },

        .subgroup_size = 8,

        .scopes = {
            .constants = ShdScopeTop,
            .gang = ShdScopeSubgroup,
            .bottom = ShdScopeInvocation,
        }
    };

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        // by default, all address spaces are physical !
        config.memory.address_spaces[i].physical = true;
        config.memory.address_spaces[i].allowed = true;
    }

    return config;
}

ArenaConfig shd_default_arena_config(const TargetConfig* target) {
    ArenaConfig config = {
        .name_bound = true,
        .allow_fold = true,
        .check_types = true,
        .validate_builtin_types = true,
        .check_op_classes = true,

        .optimisations = {
            .inline_single_use_bbs = true,
            .fold_static_control_flow = true,
            .delete_unreachable_structured_cases = true,
            .weaken_bitcast_to_lea = true,
            .assume_fixed_memory_layout = true,
        },

        .target = *target
    };

    return config;
}
