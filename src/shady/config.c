#include "shady/ir.h"
#include "shady/config.h"

#include "util.h"

#include <stdlib.h>

CompilerConfig shd_default_compiler_config(void) {
    CompilerConfig config = {

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

PtrModel get_full_physical_ptr_model(ShdIntSize ptr_size) {
    PtrModel model = {
        .ptr_size = ptr_size
    };
    for (size_t i = 0; i < NumAddressSpaces; i++) {
        model.address_spaces[i].physical = true;
        model.address_spaces[i].allowed = true;
    }
    return model;
}

ScopesLattice get_default_scopes_lattice(void) {
    return (ScopesLattice) {
        .constants = ShdScopeTop,
        .gang = ShdScopeSubgroup,
        .bottom = ShdScopeInvocation,
    };
}

MachineRules get_machine_rules_from_target_config(const TargetConfig* target) {
    MachineRules rules = {
        .ptr = target->ptr_model,
        .scopes = get_default_scopes_lattice(),
        .memory = {
            .word_size = ShdIntSize32,
            .min_align = 0,
            .fn_ptr_size = target->fn_ptr_size,
        },
        .exec_mask_size = ShdIntSize64,
    };

    // If the subgroup size is known, try to make the mask size smaller
    if (target->subgroup_size > 0) {
        if (target->subgroup_size <= 8)
            rules.exec_mask_size = ShdIntSize8;
        else if (target->subgroup_size <= 16)
            rules.exec_mask_size = ShdIntSize16;
        else if (target->subgroup_size <= 32)
            rules.exec_mask_size = ShdIntSize32;
    }

    return rules;
}

ArenaConfig shd_default_arena_config(const MachineRules* machine_rules) {
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

        .rules = *machine_rules
    };

    // arenas default to full capabilities
    config.rules.ptr = get_full_physical_ptr_model(config.rules.ptr.ptr_size);

    //if (target->capabilities.native_fncalls) {
    //    config.optimisations.weaken_non_leaking_allocas = true;
    //}

    return config;
}
