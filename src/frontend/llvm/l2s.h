#ifndef SHADY_FE_LLVM_H
#define SHADY_FE_LLVM_H

#include "shady/ir.h"
#include <stdbool.h>

typedef struct CompilerConfig_ CompilerConfig;
bool shd_parse_llvm(const CompilerConfig* config, size_t len, const char* data, String name, Module** dst);

#endif
