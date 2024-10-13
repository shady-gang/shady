#ifndef SHADY_PASSES_H

#include "shady/ir.h"
#include "shady/pass.h"

/// @name Boring, regular compiler stuff
/// @{

RewritePass shd_import;
RewritePass shd_cleanup;

/// @}

/// @name Initial CF lowering passes
/// @{

/// Gets rid of structured control flow constructs, and turns them into branches, joins and tailcalls
RewritePass shd_pass_lower_cf_instrs;
/// Uses shady.scope annotations to insert control blocks
RewritePass shd_pass_scope2control;
RewritePass shd_pass_lift_everything;
RewritePass shd_pass_remove_critical_edges;
RewritePass shd_pass_lcssa;
RewritePass shd_pass_scope_heuristic;
/// Try to identify reconvergence points throughout the program for unstructured control flow programs
RewritePass shd_pass_reconvergence_heuristics;

/// @}

/// @name Control flow lowering strategies
/// @{

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass shd_pass_lift_indirect_targets;
RewritePass shd_pass_normalize_builtins;

/// @}

/// @name Final CF lowering passes
/// @{

/// Lowers calls to stack saves and forks, lowers returns to stack pops and joins
RewritePass shd_pass_lower_callf;
/// Emulates tailcalls, forks and joins using a god function
RewritePass shd_pass_lower_tailcalls;

/// @}

/// @name Physical memory emulation
/// @{

/// Implements stack frames: saves the stack size on function entry and restores it upon exit
RewritePass shd_pass_setup_stack_frames;
/// Implements stack frames: collects allocas into a struct placed on the stack upon function entry
RewritePass shd_pass_lower_alloca;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass shd_pass_lower_stack;
/// Eliminates lea_op on all physical address spaces
RewritePass shd_pass_lower_lea;
/// Emulates generic pointers by replacing them with tagged integers and special load/store routines that look at those tags
RewritePass shd_pass_lower_generic_ptrs;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass shd_pass_lower_physical_ptrs;
/// Replaces size_of, offset_of etc with their exact values
RewritePass shd_pass_lower_memory_layout;
RewritePass shd_pass_lower_memcpy;
/// Eliminates pointers to unsized arrays from the IR. Needs lower_lea to have ran shd_first!
RewritePass shd_pass_lower_decay_ptrs;
RewritePass shd_pass_lower_generic_globals;
RewritePass shd_pass_lower_logical_pointers;

/// @}

/// @name Subgroup stuff
/// @{

/// Emulates unsupported subgroup operations using subgroup memory
RewritePass shd_pass_lower_subgroup_ops;
/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass shd_pass_lower_subgroup_vars;
/// Lowers the abstract mask type to whatever the configured target mask representation is
RewritePass shd_pass_lower_mask;

/// @}

/// @name Emulation misc.
/// @{

/// Emulates unsupported integer datatypes and operations
RewritePass shd_pass_lower_int;
RewritePass shd_pass_lower_vec_arr;
RewritePass shd_pass_lower_workgroups;
RewritePass shd_pass_lower_fill;
RewritePass shd_pass_lower_nullptr;

/// @}

/// @name Optimisation passes
/// @{

/// Eliminates all Constant decls
RewritePass shd_pass_eliminate_constants;
/// Ditto but for @Inline ones only
RewritePass shd_pass_eliminate_inlineable_constants;
/// Tags all functions that don't need special handling
RewritePass shd_pass_mark_leaf_functions;
/// In addition, also inlines function calls according to heuristics
RewritePass shd_pass_inline;
OptPass shd_opt_mem2reg;

RewritePass shd_pass_restructurize;
RewritePass shd_pass_lower_switch_btree;

RewritePass shd_pass_lower_entrypoint_args;

RewritePass shd_pass_specialize_entry_point;
RewritePass shd_pass_specialize_execution_model;

/// @}

#define SHADY_PASSES_H

#endif
