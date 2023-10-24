#ifndef SHADY_PASSES_H

#include "shady/ir.h"

typedef Module* (RewritePass)(const CompilerConfig* config, Module* src);

/// @name Boring, regular compiler stuff
/// @{

RewritePass import;

/// Removes all Unresolved nodes and replaces them with the appropriate decl/value
RewritePass bind_program;
/// Enforces the grammar, notably by let-binding any intermediary result
RewritePass normalize;
/// Makes sure every node is well-typed
RewritePass infer_program;

/// @}

/// @name Initial CF lowering passes
/// @{

/// Gets rid of structured control flow constructs, and turns them into branches, joins and tailcalls
RewritePass lower_cf_instrs;

/// @}

/// @name Control flow lowering strategies
/// @{

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass lift_indirect_targets;
/// Emulates uniform jumps within functions using a loop
RewritePass lower_jumps_loop;
/// Emulates uniform jumps within functions by applying a structuring transformation
RewritePass lower_jumps_structure;
RewritePass lcssa;
RewritePass normalize_builtins;

/// @}

/// @name Final CF lowering passes
/// @{

/// Lowers calls to stack saves and forks, lowers returns to stack pops and joins
RewritePass lower_callf;
/// Emulates tailcalls, forks and joins using a god function
RewritePass lower_tailcalls;
/// Turns SIMT code back into SIMD (intended for debugging with the help of the C backend)
RewritePass simt2d;

/// @}

/// @name Physical memory emulation
/// @{

/// Implements stack frames, saves the stack size on function entry and restores it upon exit
RewritePass setup_stack_frames;
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

/// @}

/// @name Optimisation passes
/// @{

/// Eliminates all Constant decls
RewritePass eliminate_constants;
/// Tags all functions that don't need special handling
RewritePass mark_leaf_functions;
/// Inlines basic blocks used exactly once, necessary after opt_restructure
RewritePass opt_inline_jumps;
/// In addition, also inlines function calls according to heuristics
RewritePass opt_inline;

/// Try to identify reconvergence points throughout the program for unstructured control flow programs
RewritePass reconvergence_heuristics;

RewritePass opt_stack;
RewritePass opt_restructurize;
RewritePass lower_switch_btree;

RewritePass lower_entrypoint_args;

RewritePass spirv_map_entrypoint_args;
RewritePass spirv_lift_globals_ssbo;

RewritePass specialize_entry_point;
RewritePass specialize_execution_model;

/// @}

#define SHADY_PASSES_H

#endif
