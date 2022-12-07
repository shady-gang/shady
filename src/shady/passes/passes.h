#ifndef SHADY_PASSES_H

#include "shady/ir.h"

typedef void (RewritePass)(CompilerConfig* config, Module* src_module, Module* dst_module);

// Boring, regular compiler stuff

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

// Physical memory emulation

/// Implements stack frames, saves the stack size on function entry and restores it upon exit
RewritePass setup_stack_frames;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass lower_stack;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass lower_physical_ptrs;

// Subgroup stuff

/// Emulates unsupported subgroup operations using subgroup memory
RewritePass lower_subgroup_ops;
/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass lower_subgroup_vars;
/// Lowers the abstract mask type to whatever the configured target mask representation is
RewritePass lower_mask;

// Optimisation passes

/// Eliminates all Constant decls
RewritePass eliminate_constants;
/// Tags all functions that don't need special handling
RewritePass mark_leaf_functions;
RewritePass opt_simplify_cf;
RewritePass opt_stack;
RewritePass opt_restructurize;

#define SHADY_PASSES_H

#endif
