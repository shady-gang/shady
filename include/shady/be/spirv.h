#ifndef SHD_BE_SPIRV_H
#define SHD_BE_SPIRV_H

#include "shady/ir/base.h"

typedef struct CompilerConfig_ CompilerConfig;
void emit_spirv(const CompilerConfig* config, Module*, size_t* output_size, char** output, Module** new_mod);

#endif

