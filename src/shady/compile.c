#include "ir_private.h"
#include "shady/driver.h"
#include "shady/ir.h"

#include "passes/passes.h"
#include "analysis/verify.h"

#include "../frontend/slim/parser.h"
#include "shady_scheduler_src.h"
#include "transform/internal_constants.h"

#include "util.h"
#include "log.h"

#include <stdbool.h>

static void add_scheduler_source(const CompilerConfig* config, Module* dst) {
    SlimParserConfig pconfig = {
        .front_end = true,
    };
    Module* builtin_scheduler_mod = shd_parse_slim_module(config, &pconfig, shady_scheduler_src, "builtin_scheduler");
    shd_debug_print("Adding builtin scheduler code");
    shd_module_link(dst, builtin_scheduler_mod);
    shd_destroy_ir_arena(shd_module_get_arena(builtin_scheduler_mod));
}

#ifdef NDEBUG
#define SHADY_RUN_VERIFY 0
#else
#define SHADY_RUN_VERIFY 1
#endif

void shd_run_pass_impl(const CompilerConfig* config, Module** pmod, IrArena* initial_arena, RewritePass pass, String pass_name) {
    Module* old_mod = NULL;
    old_mod = *pmod;
    *pmod = pass(config, *pmod);
    (*pmod)->sealed = true;
    shd_debugvv_print("After pass %s: \n", pass_name);
    if (SHADY_RUN_VERIFY)
        shd_verify_module(config, *pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod) && shd_module_get_arena(old_mod) != initial_arena)
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
    old_mod = *pmod;
    if (config->optimisations.cleanup.after_every_pass)
        *pmod = shd_cleanup(config, *pmod);
    shd_log_module(DEBUGVV, config, *pmod);
    if (SHADY_RUN_VERIFY)
        shd_verify_module(config, *pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod) && shd_module_get_arena(old_mod) != initial_arena)
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
    if (config->hooks.after_pass.fn)
        config->hooks.after_pass.fn(config->hooks.after_pass.uptr, pass_name, *pmod);
}

void shd_apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name) {
    bool changed = pass(config, m);
    *todo |= changed;

    if (getenv("SHADY_DUMP_CLEAN_ROUNDS") && changed) {
        shd_log_fmt(DEBUGVV, "%s changed something:\n", pass_name);
        shd_log_module(DEBUGVV, config, *m);
    }
}

CompilationResult shd_run_compiler_passes(CompilerConfig* config, Module** pmod) {
    IrArena* initial_arena = (*pmod)->arena;

    // we don't want to mess with the original module
    *pmod = shd_import(config, *pmod);
    shd_log_fmt(DEBUG, "After import:\n");
    shd_log_module(DEBUG, config, *pmod);

    if (config->input_cf.has_scope_annotations) {
        // RUN_PASS(shd_pass_scope_heuristic)
        RUN_PASS(shd_pass_lift_everything)
        RUN_PASS(shd_pass_scope2control)
    } else if (config->input_cf.restructure_with_heuristics) {
        RUN_PASS(shd_pass_remove_critical_edges)
        // RUN_PASS(shd_pass_lcssa)
        RUN_PASS(shd_pass_lift_everything)
        RUN_PASS(shd_pass_reconvergence_heuristics)
    }

    if (config->dynamic_scheduling) {
        add_scheduler_source(config, *pmod);
    }

    RUN_PASS(shd_pass_eliminate_inlineable_constants)

    RUN_PASS(shd_pass_setup_stack_frames)
    if (!config->hacks.force_join_point_lifting)
        RUN_PASS(shd_pass_mark_leaf_functions)

    RUN_PASS(shd_pass_lower_callf)
    RUN_PASS(shd_pass_inline)

    RUN_PASS(shd_pass_lift_indirect_targets)

    RUN_PASS(shd_pass_specialize_execution_model)

    //RUN_PASS(shd_pass_opt_stack)

    if (config->specialization.entry_point)
        RUN_PASS(shd_pass_specialize_entry_point)

    RUN_PASS(shd_pass_add_init_fini)
    RUN_PASS(shd_pass_promote_io_variables)

    RUN_PASS(shd_pass_lower_tailcalls)
    //RUN_PASS(shd_pass_lower_switch_btree)
    //RUN_PASS(shd_pass_opt_mem2reg)

    RUN_PASS(shd_pass_lower_logical_pointers)

    RUN_PASS(shd_pass_lower_mask)
    RUN_PASS(shd_pass_lower_subgroup_ops)
    if (config->lower.emulate_physical_memory) {
        RUN_PASS(shd_pass_lower_alloca)
    }
    RUN_PASS(shd_pass_lower_stack)
    RUN_PASS(shd_pass_lower_memcpy)
    RUN_PASS(shd_pass_lower_lea)
    RUN_PASS(shd_pass_lower_generic_globals)
    if (config->lower.emulate_generic_ptrs) {
        RUN_PASS(shd_pass_lower_generic_ptrs)
    }
    if (config->lower.emulate_physical_memory) {
        RUN_PASS(shd_pass_lower_physical_ptrs)
    }
    RUN_PASS(shd_pass_lower_subgroup_vars)
    RUN_PASS(shd_pass_lower_memory_layout)

    if (config->lower.decay_ptrs)
        RUN_PASS(shd_pass_lower_decay_ptrs)

    RUN_PASS(shd_pass_lower_int)

    RUN_PASS(shd_pass_lower_fill)
    RUN_PASS(shd_pass_lower_nullptr)
    RUN_PASS(shd_pass_normalize_builtins)

    RUN_PASS(shd_pass_restructurize)

    return CompilationNoError;
}

#undef mod
