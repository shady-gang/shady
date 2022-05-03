#ifndef SHADY_PASSES_H

#include "shady/ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);

/// Rewrites a whole program, starting at the root
typedef const Node* (RewritePass)(IrArena* src_arena, IrArena* dst_arena, const Node* src_root);

RewritePass bind_program;
RewritePass type_program;
RewritePass lower_cf_instrs;
RewritePass lower_callf;

#define SHADY_PASSES_H

#endif
