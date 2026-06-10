#include "shady/ir.h"

#include "util.h"

static void add_logical_ptr_limitations(TargetConfig* target_config) {
    // by default everything is logical
    for (int i = 0; i < NumAddressSpaces; i++)
        target_config->ptr_model.address_spaces[(AddressSpace) i].physical = false;
    // no generic pointers either
    target_config->ptr_model.address_spaces[AsGeneric].allowed = false;
}

static void add_default_shading_language_limitations(TargetConfig* target_config) {
    add_logical_ptr_limitations(target_config);
    target_config->capabilities.native_fncalls = false;
    target_config->capabilities.native_tailcalls = false;
    target_config->capabilities.native_memcpy = false;
    target_config->capabilities.native_stack = false;
    target_config->capabilities.linkage = false;
}

TargetConfig shd_default_target_config(void) {
    TargetConfig config = {
        .ptr_model = get_full_physical_ptr_model(ShdIntSize64),

        .fn_ptr_size = ShdIntSize64,

        .subgroup_size = 0,

        .capabilities = {
            .native_fncalls = true,
            .native_tailcalls = true,

            .linkage = true,
            .maximal_reconvergence = true,
        },
    };

    return config;
}

void shd_target_configure_defaults_for_arch(TargetConfig* target_config) {
    switch (target_config->arch) {
        case TgtNone: /* no target */  break;
        case TgtSPV:
            // Default to 64-wide to support GCN cards.
            target_config->subgroup_size = 64;
            add_default_shading_language_limitations(target_config);
            // default to assuming BDA support on Vulkan
            target_config->ptr_model.address_spaces[AsGlobal].physical = true;
            target_config->fn_ptr_size = ShdIntSize32;

            // if (compiler_config->use_rt_pipelines_for_calls) {
            //     target_config->capabilities.native_fncalls = true;
            //     target_config->capabilities.rt_pipelines = true;
            //     target_config->fn_ptr_size = ShdIntSize32;
            // }
            break;
        case TgtC:
            break;
        case TgtGLSL:
            add_default_shading_language_limitations(target_config);
            break;
        case TgtISPC:
            target_config->subgroup_size = 8;
            add_default_shading_language_limitations(target_config);
            for (size_t i = 0; i < NumAddressSpaces; i++) {
                if (i != AsGeneric && shd_get_addr_space_scope(i) < ShdScopeSubgroup) {
                    // ISPC can use native physical pointers for `uniform` data
                    // Due to how it lays out types for `varying` data, we want to emulate memory ourselves.
                    target_config->ptr_model.address_spaces[AsGlobal].physical = true;
                }
            }
            break;
        case TgtCUDA:
            target_config->subgroup_size = 32;
            //target_config->memory.fn_ptr_size = IntTy64;
            //target_config->memory.word_size = IntTy8;
            //target_config->capabilities.native_stack = true;
            //target_config->capabilities.native_memcpy = true;
            //target_config->memory.max_align = 8;
            add_default_shading_language_limitations(target_config);
            target_config->capabilities.linkage = true;
            target_config->ptr_model.address_spaces[AsGlobal].physical = true;
            break;
    }
}

void shd_target_apply_execution_model_restrictions(TargetConfig* target, ShdExecutionModel execution_model) {
    switch (execution_model) {
        case ShdExecutionModelVertex:
        case ShdExecutionModelFragment: {
            target->ptr_model.address_spaces[AsShared].allowed = false;
        }
        default: break;
    }

    if (!target->ptr_model.address_spaces[AsShared].allowed)
        target->ptr_model.address_spaces[AsSubgroup].allowed = false;
}

