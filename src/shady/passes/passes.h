#ifndef SHADY_PASSES_H

#include "shady/ir.h"

#include <stdio.h>

/// Rewrites a whole program, starting at the root
typedef const Node* (RewritePass)(CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_root);

RewritePass bind_program;
RewritePass normalize;
RewritePass infer_program;
/// Gets rid of structured control flow constructs, and turns them into branches, joins and callcs
RewritePass lower_cf_instrs;
/// Extracts non-structured CF targets (continuations) inside basic blocks into separate functions (including spilling)
RewritePass lower_continuations;
/// Lowers calls to stack saves and branches, lowers returns to stack pops and joins
RewritePass lower_callf;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass lower_stack;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass lower_physical_ptrs;

RewritePass setup_stack_frames;
RewritePass eliminate_constants;
RewritePass lower_subgroup_ops;

RewritePass lower_mask;

// Optimisation passes

/// Eliminates indirect calls where legal (the address isn't being captured, and there is no recursion)
RewritePass remove_indirect_calls;
RewritePass opt_simplify_cf;
RewritePass opt_restructurize;

// Control flow lowering strategies
/// Emulates branches and joins using a god function
RewritePass lower_tailcalls;
/// Emulates uniform jumps within functions using a loop
RewritePass lower_jumps_loop;
/// Emulates uniform jumps within functions by applying a structuring transformation
RewritePass lower_jumps_structure;

#define SHADY_PASSES_H

#endif
