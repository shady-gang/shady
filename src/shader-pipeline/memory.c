#include "shader_pipeline.h"

/// Implements stack frames: collects allocas into a struct placed on the stack upon function entry
RewritePass shd_pass_lower_alloca;
/// Turns stack pushes and pops into accesses into pointer load and stores
RewritePass shd_pass_lower_stack_access;
/// Eliminates lea_op on all physical address spaces
RewritePass shd_pass_lower_lea;
/// Emulates generic pointers by replacing them with tagged integers and special load/store routines that look at those tags
RewritePass shd_pass_lower_generic_ptrs;
/// Emulates physical pointers to certain address spaces by using integer indices into global arrays
RewritePass shd_pass_lower_physical_memory;
/// Replaces size_of, offset_of etc with their exact values
RewritePass shd_pass_lower_memory_layout;
RewritePass shd_pass_lower_memcpy;
/// Eliminates pointers to unsized arrays from the IR. Needs lower_lea to have ran shd_first!
RewritePass shd_pass_lower_decay_ptrs;
RewritePass shd_pass_lower_logical_pointers;
RewritePass shd_pass_promote_io_variables;

/// Lowers subgroup logical variables into something that actually exists (likely a carved out portion of shared memory)
RewritePass shd_pass_lower_subgroup_vars;

static void lower_memory(TargetConfig* target, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_promote_io_variables, config)
    RUN_PASS(shd_pass_lower_logical_pointers, config)

    if (!target->capabilities.native_memcpy) {
        RUN_PASS(shd_pass_lower_memcpy, config)
    }

    if (!target->capabilities.native_stack) {
        RUN_PASS(shd_pass_lower_alloca, config)
        RUN_PASS(shd_pass_lower_stack_access, config)
    }
    RUN_PASS(shd_pass_lower_lea, target)
    if (!target->memory.address_spaces[AsGeneric].allowed) {
        RUN_PASS(shd_pass_lower_generic_ptrs, config)
    }
    RUN_PASS(shd_pass_lower_physical_memory, target)
    RUN_PASS(shd_pass_lower_subgroup_vars, config)
    RUN_PASS(shd_pass_lower_memory_layout, config)
    if (config->lower.decay_ptrs)
        RUN_PASS(shd_pass_lower_decay_ptrs, config)
}

void shd_pipeline_add_memory_lowering(ShdPipeline pipeline, TargetConfig tgt) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) lower_memory, &tgt, sizeof(tgt));
}
