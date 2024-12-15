#ifndef SHD_BE_C_H
#define SHD_BE_C_H

#include "shady/ir/base.h"

typedef struct ShdPipeline_* ShdPipeline;

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
    int glsl_version;
} CTargetConfig;

CTargetConfig shd_default_c_target_config(void);

typedef struct CompilerConfig_ CompilerConfig;

void shd_pipeline_add_c_target_passes(ShdPipeline pipeline, CTargetConfig* econfig);
void shd_emit_c(const CompilerConfig* compiler_config, CTargetConfig target_config, Module* mod, size_t* output_size, char** output);

#endif

