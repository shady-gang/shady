#ifndef SHADY_PASSES_H

#include "shady/ir.h"
#include "shady/pass.h"

/// @name Boring, regular compiler stuff
/// @{

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

RewritePass shd_pass_add_init_fini;

/// @}

/// @name Control flow lowering strategies
/// @{

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass shd_pass_lift_indirect_targets;

/// @}

/// @name Final CF lowering passes
/// @{


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
RewritePass shd_pass_promote_io_variables;

/// @}

/// @name Subgroup stuff
/// @{
///
/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass shd_pass_lower_subgroup_vars;
/// Lowers the abstract mask type to whatever the configured target mask representation is
RewritePass shd_pass_lower_mask;

/// @}

/// @name Emulation misc.
/// @{

RewritePass shd_pass_lower_vec_arr;
RewritePass shd_pass_lower_workgroups;
RewritePass shd_pass_lower_inclusive_scan;

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

RewritePass shd_pass_specialize_entry_point;
RewritePass shd_pass_specialize_execution_model;

/// @}

#define SHADY_PASSES_H

#endif
