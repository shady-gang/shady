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

        .optimisations = {
            .cleanup = {
                .after_every_pass = true,
                .delete_unused_instructions = true,
            }
        },
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

    String max_top_iterations = getenv("SHADY_MAX_TOP_ITERATIONS");
    if (max_top_iterations) {
        config.shader_diagnostics.max_top_iterations = strtoll(max_top_iterations, NULL, 10);
    }

    return config;
}

TargetConfig shd_default_target_config(void) {
    TargetConfig config = {
        .memory = {
            .word_size = ShdIntSize32,
            .ptr_size = ShdIntSize64,
            .fn_ptr_size = ShdIntSize64,
            .exec_mask_size = ShdIntSize64
        },

        .subgroup_size = 0,

        .scopes = {
            .constants = ShdScopeTop,
            .gang = ShdScopeSubgroup,
            .bottom = ShdScopeInvocation,
        },

        .capabilities = {
            .native_fncalls = true,
            .native_tailcalls = true,

            .linkage = true,
        }
    };

    for (size_t i = 0; i < NumAddressSpaces; i++) {
        // by default, all address spaces are physical !
        config.memory.address_spaces[i].physical = true;
        config.memory.address_spaces[i].allowed = true;
    }

    return config;
}

void shd_target_apply_execution_model_restrictions(TargetConfig* target) {
    switch (target->execution_model) {
        case ShdExecutionModelVertex:
        case ShdExecutionModelFragment: {
            target->memory.address_spaces[AsShared].allowed = false;
        }
        default: break;
    }

    if (!target->memory.address_spaces[AsShared].allowed)
        target->memory.address_spaces[AsSubgroup].allowed = false;
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

    TargetConfig default_target = shd_default_target_config();

    // arenas default to full capabilities
    memcpy(&config.target.capabilities, &default_target.capabilities, sizeof(target->capabilities));
    memcpy(&config.target.memory.address_spaces, &default_target.memory.address_spaces, sizeof(target->memory.address_spaces));

    if (target->capabilities.native_fncalls) {
        config.optimisations.weaken_non_leaking_allocas = true;
    }

    return config;
}
