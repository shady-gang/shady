#ifndef SHADY_FE_LLVM_H
#define SHADY_FE_LLVM_H

#include "shady/ir.h"

#include <stdbool.h>

bool shd_parse_llvm(const CompilerConfig*, const TargetConfig*, size_t len, const char* data, String name, Module** dst);

int shd_get_linked_major_llvm_version();

#endif
