#ifndef SHADY_SHADER_PIPELINE_H
#define SHADY_SHADER_PIPELINE_H

#include "shady/pipeline/pipeline.h"

typedef enum {
    FCL_None,
    FCL_SoftwareScheduler,
    FCL_RT_Callables,
} FunctionCallLowering;

typedef struct {
    //const TargetConfig* target;
    const ExecutionModelInfo* exec_model_info;

    FunctionCallLowering function_call_lowering;
    uint32_t per_thread_stack_size;
} ShaderLoweringConfig;

ShaderLoweringConfig shd_default_shader_target_config(void);

void shd_parse_shader_target_config_args(ShaderLoweringConfig* config, int* pargc, char** argv);

void shd_pipeline_add_shader_target_lowering(ShdPipeline pipeline, const ShaderLoweringConfig* config, const TargetConfig* target);

#endif