#ifndef SHD_BE_C_H
#define SHD_BE_C_H

#include "shady/ir.h"

typedef enum {
    CDialect_C11,
    CDialect_GLSL,
    CDialect_ISPC,
    CDialect_CUDA,
} CDialect;

typedef struct {
    CDialect dialect;
    bool explicitly_sized_types;
    bool allow_compound_literals;
    bool decay_unsized_arrays;
} CEmitterConfig;

typedef struct CompilerConfig_ CompilerConfig;
void emit_c(const CompilerConfig* compiler_config, CEmitterConfig emitter_config, Module*, size_t* output_size, char** output, Module** new_mod);

#endif

