#include "shader_pipeline.h"
#include "shady/pass.h"

#include "portability.h"
#include "log.h"

void shd_pipeline_add_feature_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_memory_lowering(pipeline, tgt);
    shd_pipeline_add_polyfills(pipeline, tgt);
    shd_pipeline_add_restructure_cf(pipeline);
}

void shd_pipeline_add_restructure_cf(ShdPipeline pipeline);

static Module* specialize_target_config(SHADY_UNUSED const CompilerConfig* config, TargetConfig* target_config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.target.subgroup_size = target_config->subgroup_size;
    //specialize_arena_config(*em, &aconfig.target);

    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Rewriter r = shd_create_importer(src, dst);
    shd_rewrite_module(&r);
    shd_destroy_rewriter(&r);
    return dst;
}

static CompilationResult specialize_target_config_step(TargetConfig* target_config, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(specialize_target_config, (void*) target_config);
    return CompilationNoError;
}

void shd_pipeline_add_target_specialization(ShdPipeline pipeline, TargetConfig target_config) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) specialize_target_config_step, &target_config, sizeof(TargetConfig));
}

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_target_specialization(pipeline, tgt);
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