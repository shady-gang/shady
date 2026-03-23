#include "shady/pipeline/shader_pipeline.h"
#include "shady/pass.h"

void shd_pipeline_add_target_specialization(ShdPipeline pipeline, const TargetConfig* em);
void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ShdExecutionModel em);
void shd_pipeline_add_specialize_entry_point(ShdPipeline pipeline, String entry_point);

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, const ShaderLoweringConfig* tgt, const TargetConfig*, uint32_t);
void shd_pipeline_add_polyfills(ShdPipeline pipeline, const ShaderLoweringConfig* tgt);

void shd_pipeline_add_fncall_emulation(ShdPipeline pipeline, const ShaderLoweringConfig*, const TargetConfig*, uint32_t);
void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);