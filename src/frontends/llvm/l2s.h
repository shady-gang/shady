#ifndef SHADY_FE_LLVM_H
#define SHADY_FE_LLVM_H

#include "shady/ir.h"
#include <stdbool.h>

bool parse_llvm_into_shady(Module* dst, size_t len, const char* data);

#endif
