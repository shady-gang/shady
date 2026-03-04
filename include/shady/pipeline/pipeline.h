#ifndef SHD_PIPELINE_H
#define SHD_PIPELINE_H

#include "shady/ir/base.h"
#include "shady/config.h"

typedef struct ShdPipeline_* ShdPipeline;

ShdPipeline shd_create_empty_pipeline(void);
void shd_destroy_pipeline(ShdPipeline);

/// Runs a given pipeline on a module
ShdResult shd_pipeline_run(ShdPipeline, const CompilerConfig* config, Module** pmod);

typedef ShdResult (*ShdPipelineStepFn)(void*, const CompilerConfig*, Module**);

void shd_pipeline_add_step(ShdPipeline, ShdPipelineStepFn, void* payload, size_t payload_size);

#endif
