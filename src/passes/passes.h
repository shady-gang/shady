#ifndef SHADY_PASSES_H

#include "shady/ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);

RewritePass bind_program;
RewritePass type_program;
RewritePass instr2bb;

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
