#include "shady/pipeline/pipeline.h"
#include "shady/pass.h"

void shd_pipeline_add_specialize_execution_model(ShdPipeline pipeline, ExecutionModel em);
void shd_pipeline_add_specialize_entry_point(ShdPipeline pipeline, String entry_point);
void shd_pipeline_add_init_fini(ShdPipeline pipeline);

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, TargetConfig tgt);
void shd_pipeline_add_polyfills(ShdPipeline pipeline, TargetConfig tgt);

void shd_pipeline_add_remove_indirect_calls(ShdPipeline pipeline);
void shd_pipeline_add_tailcall_elimination(ShdPipeline pipeline);
void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);