#ifndef SHADY_PASSES_H

#include "ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);
void emit(IrArena*, const Node* root, FILE* output);
const Node* instr2bb(IrArena* src_arena, IrArena* dst_arena, const Node* src_program);

RewritePass bind_program;
RewritePass type_program;

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
