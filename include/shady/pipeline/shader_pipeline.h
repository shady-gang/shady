#ifndef SHADY_SHADER_PIPELINE_H
#define SHADY_SHADER_PIPELINE_H

#include "shady/pipeline/pipeline.h"

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, const TargetConfig tgt, CompilerConfig* hacky_bs);

#endif