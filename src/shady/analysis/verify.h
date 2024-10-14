#ifndef SHADY_VERIFY_H
#define SHADY_VERIFY_H

#include "shady/ir.h"

typedef struct CompilerConfig_ CompilerConfig;
void shd_verify_module(const CompilerConfig* config, Module* mod);

#endif
