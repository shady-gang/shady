#ifndef SHADY_PASSES_H

#include "shady/ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);

/// Rewrites a whole program, starting at the root
typedef const Node* (RewritePass)(CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_root);

RewritePass bind_program;
RewritePass infer_program;
/// Gets rid of structured control flow constructs, and turns them into jumps, branches and callc
RewritePass lower_cf_instrs;
/// Turns callc into callf by extracting the return continuations into dedicated functions
RewritePass lower_callc;
/// Emulates function calls using the stack and a big outer loop
RewritePass lower_callf;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass lower_stack;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass lower_physical_ptrs;
/// Emulates uniform jumps within functions using a loop
RewritePass lower_jumps_loop;
/// Emulates uniform jumps within functions by applying a structuring transformation
RewritePass lower_jumps_structure;

#define SHADY_PASSES_H

#endif
