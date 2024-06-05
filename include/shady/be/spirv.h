#ifndef SHD_BE_SPIRV_H
#define SHD_BE_SPIRV_H

#include "shady/ir.h"

void emit_spirv(CompilerConfig* config, Module*, size_t* output_size, char** output, Module** new_mod);

#endif

