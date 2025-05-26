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
        bool maximal_reconvergence;
    } features;

    struct {
        bool shuffle_instead_of_broadcast_first;
        bool avoid_spirv_cross_broken_bda_pointers;
    } hacks;
} SPVBackendConfig;

SPVBackendConfig shd_default_spirv_backend_config(void);

void shd_pipeline_add_spirv_target_passes(ShdPipeline, const TargetConfig*, const SPVBackendConfig*);
void shd_emit_spirv(const CompilerConfig*, SPVBackendConfig, Module*, size_t* output_size, char** output);

#endif

