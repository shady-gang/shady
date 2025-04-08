#include "pipeline/pipeline_private.h"

#include "log.h"

void shd_pipeline_add_feature_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_memory_lowering(pipeline, tgt);
    shd_pipeline_add_polyfills(pipeline, tgt);
    shd_pipeline_add_restructure_cf(pipeline);
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    if (tgt.execution_model != EmNone)
        shd_pipeline_add_specialize_execution_model(pipeline, tgt.execution_model);
    if (tgt.entry_point) {
        if (tgt.execution_model == EmNone)
            shd_log_fmt(WARN, "Specializing on an entry point but no execution model picked!");
        shd_pipeline_add_specialize_entry_point(pipeline, tgt.entry_point);
    }

    shd_pipeline_add_fncall_emulation(pipeline, tgt);
    shd_pipeline_add_feature_lowering(pipeline, tgt);
}