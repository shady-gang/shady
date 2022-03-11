#ifndef SHADY_PASSES_H

#include "ir.h"

#include <stdio.h>

const Node* parse(char* contents, IrArena* arena);
void emit(const Node* root, FILE* output);

RewritePass bind_program;
RewritePass type_program;

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
