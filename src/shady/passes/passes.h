#ifndef SHADY_PASSES_H

#include "shady/ir.h"
#include "shady/pass.h"

/// @name Boring, regular compiler stuff
/// @{

RewritePass import;
RewritePass cleanup;

/// @}

/// @name Initial CF lowering passes
/// @{

/// Gets rid of structured control flow constructs, and turns them into branches, joins and tailcalls
RewritePass lower_cf_instrs;
/// Uses shady.scope annotations to insert control blocks
RewritePass scope2control;
RewritePass lift_everything;
RewritePass lcssa;
RewritePass scope_heuristic;
/// Try to identify reconvergence points throughout the program for unstructured control flow programs
RewritePass reconvergence_heuristics;

/// @}

/// @name Control flow lowering strategies
/// @{

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass lift_indirect_targets;
RewritePass normalize_builtins;

/// @}

/// @name Final CF lowering passes
/// @{

/// Lowers calls to stack saves and forks, lowers returns to stack pops and joins
RewritePass lower_callf;
/// Emulates tailcalls, forks and joins using a god function
RewritePass lower_tailcalls;

/// @}

/// @name Physical memory emulation
/// @{

/// Implements stack frames: saves the stack size on function entry and restores it upon exit
RewritePass setup_stack_frames;
/// Implements stack frames: collects allocas into a struct placed on the stack upon function entry
RewritePass lower_alloca;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass lower_stack;
/// Eliminates lea_op on all physical address spaces
RewritePass lower_lea;
/// Emulates generic pointers by replacing them with tagged integers and special load/store routines that look at those tags
RewritePass lower_generic_ptrs;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass lower_physical_ptrs;
/// Replaces size_of, offset_of etc with their exact values
RewritePass lower_memory_layout;
RewritePass lower_memcpy;
/// Eliminates pointers to unsized arrays from the IR. Needs lower_lea to have ran first!
RewritePass lower_decay_ptrs;
RewritePass lower_generic_globals;
RewritePass lower_logical_pointers;

/// @}

/// @name Subgroup stuff
/// @{

/// Emulates unsupported subgroup operations using subgroup memory
RewritePass lower_subgroup_ops;
/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass lower_subgroup_vars;
/// Lowers the abstract mask type to whatever the configured target mask representation is
RewritePass lower_mask;

/// @}

/// @name Emulation misc.
/// @{

/// Emulates unsupported integer datatypes and operations
RewritePass lower_int;
RewritePass lower_vec_arr;
RewritePass lower_workgroups;
RewritePass lower_fill;
RewritePass lower_nullptr;

/// @}

/// @name Optimisation passes
/// @{

/// Eliminates all Constant decls
RewritePass eliminate_constants;
/// Ditto but for @Inline ones only
RewritePass eliminate_inlineable_constants;
/// Tags all functions that don't need special handling
RewritePass mark_leaf_functions;
/// In addition, also inlines function calls according to heuristics
RewritePass opt_inline;
OptPass opt_mem2reg;

RewritePass opt_stack;
RewritePass opt_restructurize;
RewritePass lower_switch_btree;

RewritePass lower_entrypoint_args;

RewritePass specialize_entry_point;
RewritePass specialize_execution_model;

/// @}

#define SHADY_PASSES_H

#endif
