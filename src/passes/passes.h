#ifndef SHADY_PASSES_H

#include "shady/ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);

/// Rewrites a whole program, starting at the root
typedef const Node* (RewritePass)(CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_root);

RewritePass bind_program;
RewritePass type_program;
/// Gets rid of structured control flow constructs, and turns them into jumps, branches and callc
RewritePass lower_cf_instrs;
/// Turns callc into callf by extracting the return continuations into dedicated functions
RewritePass lower_callc;
/// Emulates function calls using the stack
RewritePass lower_callf;
RewritePass lower_stack;

#define SHADY_PASSES_H

#endif
