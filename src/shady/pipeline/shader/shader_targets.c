#include "pipeline/pipeline_private.h"

#include "log.h"

void shd_pipeline_add_feature_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_memory_lowering(pipeline, tgt);
    shd_pipeline_add_polyfills(pipeline, tgt);
    shd_pipeline_add_restructure_cf(pipeline);
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, TargetConfig tgt, ExecutionModel em, String entry_point) {
    if (em != EmNone)
        shd_pipeline_add_specialize_execution_model(pipeline, em);
    if (entry_point && em == EmNone)
        shd_error("Specializing on an entry point but no execution model picked!");
    if (entry_point) {
        shd_pipeline_add_specialize_entry_point(pipeline, entry_point);
    }

    shd_pipeline_add_fncall_emulation(pipeline);
    shd_pipeline_add_feature_lowering(pipeline, tgt);
}