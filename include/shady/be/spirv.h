#ifndef SHD_BE_SPIRV_H
#define SHD_BE_SPIRV_H

#include "shady/ir/base.h"

typedef struct ShdPipeline_* ShdPipeline;

typedef struct CompilerConfig_ CompilerConfig;

typedef struct {
    struct {
        uint8_t major;
        uint8_t minor;
    } target_version;

    struct {
        bool shuffle_instead_of_broadcast_first;
    } hacks;
} SPIRVTargetConfig;

SPIRVTargetConfig shd_default_spirv_target_config(void);

void shd_pipeline_add_spirv_target_passes(ShdPipeline pipeline, SPIRVTargetConfig* econfig);
void shd_emit_spirv(const CompilerConfig* config, SPIRVTargetConfig target_config, Module* mod, size_t* output_size, char** output);

#endif

