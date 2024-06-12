#include "shady/driver.h"
#include "compile.h"

#include "frontends/slim/parser.h"
#include "shady_scheduler_src.h"
#include "transform/internal_constants.h"
#include "portability.h"
#include "ir_private.h"
#include "util.h"

#include <stdbool.h>

void add_scheduler_source(const CompilerConfig* config, Module* dst) {
    ParserConfig pconfig = {
        .front_end = true,
    };
    Module* builtin_scheduler_mod = parse_slim_module(config, pconfig, shady_scheduler_src, "builtin_scheduler");
    debug_print("Adding builtin scheduler code");
    link_module(dst, builtin_scheduler_mod);
    destroy_ir_arena(get_module_arena(builtin_scheduler_mod));
}

void run_pass_impl(const CompilerConfig* config, Module** pmod, IrArena* initial_arena, RewritePass pass, String pass_name) {
    Module* old_mod = NULL;
    old_mod = *pmod;
    *pmod = pass(config, *pmod);
    (*pmod)->sealed = true;
    if (SHADY_RUN_VERIFY)
        verify_module(config, *pmod);
    if (get_module_arena(old_mod) != get_module_arena(*pmod) && get_module_arena(old_mod) != initial_arena)
        destroy_ir_arena(get_module_arena(old_mod));
    old_mod = *pmod;
    if (config->optimisations.cleanup.after_every_pass)
        *pmod = cleanup(config, *pmod);
    debugvv_print("After pass %s: \n", pass_name);
    log_module(DEBUGVV, config, *pmod);
    if (SHADY_RUN_VERIFY)
        verify_module(config, *pmod);
    if (get_module_arena(old_mod) != get_module_arena(*pmod) && get_module_arena(old_mod) != initial_arena)
        destroy_ir_arena(get_module_arena(old_mod));
    if (config->hooks.after_pass.fn)
        config->hooks.after_pass.fn(config->hooks.after_pass.uptr, pass_name, *pmod);
}

CompilationResult run_compiler_passes(CompilerConfig* config, Module** pmod) {
    IrArena* initial_arena = (*pmod)->arena;
	
    if (config->dynamic_scheduling) {
		*pmod = import(config, *pmod); // we don't want to mess with the original module
	
        add_scheduler_source(config, *pmod);
	}

    RUN_PASS(eliminate_inlineable_constants)	
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

    RUN_PASS(specialize_execution_model)

    //RUN_PASS(opt_stack)

    RUN_PASS(lower_tailcalls)
    RUN_PASS(lower_switch_btree)
    RUN_PASS(opt_restructurize)
    RUN_PASS(opt_mem2reg)

    if (config->specialization.entry_point)
        RUN_PASS(specialize_entry_point)

    RUN_PASS(lower_logical_pointers)

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
    RUN_PASS(lower_nullptr)
    RUN_PASS(normalize_builtins)

    return CompilationNoError;
}

#undef mod
