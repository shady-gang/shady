#ifndef SHADY_VERIFY_H
#define SHADY_VERIFY_H

#include "shady/ir.h"

typedef struct CompilerConfig_ CompilerConfig;
void verify_module(const CompilerConfig*, Module*);

#endif
