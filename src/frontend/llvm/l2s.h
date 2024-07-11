#ifndef SHADY_FE_LLVM_H
#define SHADY_FE_LLVM_H

#include "shady/ir.h"
#include <stdbool.h>

typedef struct CompilerConfig_ CompilerConfig;
bool parse_llvm_into_shady(const CompilerConfig*, size_t len, const char* data, String name, Module** dst);

#endif
