#include "shady/pipeline/shader_pipeline.h"
#include "shady/pass.h"

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, TargetConfig tgt, CompilerConfig* hacky_bs);

void shd_pipeline_add_target_specialization(ShdPipeline pipeline, TargetConfig em);
void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ShdExecutionModel em);
void shd_pipeline_add_specialize_entry_point(ShdPipeline pipeline, String entry_point);

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, TargetConfig tgt);
void shd_pipeline_add_polyfills(ShdPipeline pipeline, TargetConfig tgt);

void shd_pipeline_add_fncall_emulation(ShdPipeline pipeline, TargetConfig);
void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);