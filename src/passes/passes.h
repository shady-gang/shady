#ifndef SHADY_PASSES_H

#include "ir.h"

#include <stdio.h>

const struct Node* parse(char* contents, struct IrArena* arena);
void emit(const struct Node* root, FILE* output);

RewritePass bind_program;
RewritePass type_program;

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
