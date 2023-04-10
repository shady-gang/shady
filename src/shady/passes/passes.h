#ifndef SHADY_PASSES_H

#include "shady/ir.h"

typedef void (RewritePass)(CompilerConfig* config, Module* src_module, Module* dst_module);

// Boring, regular compiler stuff

RewritePass import;

/// Removes all Unresolved nodes and replaces them with the appropriate decl/value
RewritePass bind_program;
/// Enforces the grammar, notably by let-binding any intermediary result
RewritePass normalize;
/// Makes sure every node is well-typed
RewritePass infer_program;

// Initial CF lowering passes

/// Gets rid of structured control flow constructs, and turns them into branches, joins and tailcalls
RewritePass lower_cf_instrs;

// Control flow lowering strategies

/// Extracts unstructured basic blocks into separate functions (including spilling)
RewritePass lower_continuations;
/// Emulates uniform jumps within functions using a loop
RewritePass lower_jumps_loop;
/// Emulates uniform jumps within functions by applying a structuring transformation
RewritePass lower_jumps_structure;

// Final CF lowering passes

/// Lowers calls to stack saves and forks, lowers returns to stack pops and joins
RewritePass lower_callf;
/// Emulates tailcalls, forks and joins using a god function
RewritePass lower_tailcalls;
/// Turns SIMT code back into SIMD (intended for debugging with the help of the C backend)
RewritePass simt2d;

// Physical memory emulation

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

// Subgroup stuff

/// Emulates unsupported subgroup operations using subgroup memory
RewritePass lower_subgroup_ops;
/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass lower_subgroup_vars;
/// Lowers the abstract mask type to whatever the configured target mask representation is
RewritePass lower_mask;

// Emulation misc.

/// Emulates unsupported integer datatypes and operations
RewritePass lower_int;
RewritePass lower_vec_arr;
RewritePass lower_workgroups;
RewritePass lower_fill;

// Optimisation passes

/// Eliminates all Constant decls
RewritePass eliminate_constants;
/// Tags all functions that don't need special handling
RewritePass mark_leaf_functions;
RewritePass opt_simplify_cf;
RewritePass opt_stack;
RewritePass opt_restructurize;

RewritePass lower_entrypoint_args;

RewritePass spirv_map_entrypoint_args;
RewritePass specialize_for_entry_point;
RewritePass spirv_lift_globals_ssbo;

#define SHADY_PASSES_H

#endif
