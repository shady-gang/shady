#include "emit_spv.h"

#include "shady/ir/builtin.h"

#include "../shady/ir_private.h"
#include "../shady/analysis/cfg.h"
#include "../shady/passes/passes.h"
#include "../shady/analysis/scheduler.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "growy.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>

SPIRVTargetConfig shd_default_spirv_target_config(void) {
    SPIRVTargetConfig config = {
        .target_version = {
            .major = 1,
            .minor = 4
        },
    };
    return config;
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

#pragma GCC diagnostic error "-Wswitch"

void spv_register_emitted(Emitter* emitter, FnBuilder* fn_builder, const Node* node, SpvId id) {
    if (is_value(node)) {
        String name = shd_get_node_name_unsafe(node);
        if (name)
            spvb_name(emitter->file_builder, id, name);
    }
    struct Dict* map = fn_builder ? fn_builder->emitted : emitter->global_node_ids;
    shd_dict_insert_get_result(struct Node*, SpvId, map, node, id);
}

SpvId* spv_search_emitted(Emitter* emitter, FnBuilder* fn_builder, const Node* node) {
    SpvId* found = NULL;
    if (fn_builder)
        found = shd_dict_find_value(const Node*, SpvId, fn_builder->emitted, node);
    if (!found)
        found = shd_dict_find_value(const Node*, SpvId, emitter->global_node_ids, node);
    return found;
}

SpvId spv_find_emitted(Emitter* emitter, FnBuilder* fn_builder, const Node* node) {
    SpvId* found = spv_search_emitted(emitter, fn_builder, node);
    return *found;
}

static void emit_basic_block(Emitter* emitter, FnBuilder* fn_builder, const CFNode* cf_node) {
    const Node* bb_node = cf_node->node;
    assert(is_basic_block(bb_node) || cf_node == fn_builder->cfg->entry);

    const Node* body = get_abstraction_body(bb_node);

    // Find the preassigned ID to this
    BBBuilder bb_builder = spv_find_basic_block_builder(emitter, bb_node);
    SpvId bb_id = spvb_get_block_builder_id(bb_builder);
    spvb_add_bb(fn_builder->base, bb_builder);

    String name = shd_get_abstraction_name_safe(bb_node);
    if (name)
        spvb_name(emitter->file_builder, bb_id, name);

    spv_emit_terminator(emitter, fn_builder, bb_builder, bb_node, body);

    for (size_t i = 0; i < shd_list_count(cf_node->dominates); i++) {
        CFNode* dominated = shd_read_list(CFNode*, cf_node->dominates)[i];
        emit_basic_block(emitter, fn_builder, dominated);
    }

    if (fn_builder->per_bb[cf_node->rpo_index].continue_builder)
        spvb_add_bb(fn_builder->base, fn_builder->per_bb[cf_node->rpo_index].continue_builder);
}

static void emit_function(Emitter* emitter, const Node* node) {
    assert(node->tag == Function_TAG);

    const Type* fn_type = node->type;
    SpvId fn_id = spv_find_emitted(emitter, NULL, node);
    FnBuilder fn_builder = {
        .base = spvb_begin_fn(emitter->file_builder, fn_id, spv_emit_type(emitter, fn_type), spv_types_to_codom(emitter, node->payload.fun.return_types)),
        .emitted = shd_new_dict(Node*, SpvId, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .cfg = build_fn_cfg(node),
    };
    fn_builder.scheduler = shd_new_scheduler(fn_builder.cfg);
    fn_builder.per_bb = calloc(sizeof(*fn_builder.per_bb), fn_builder.cfg->size);

    Nodes params = node->payload.fun.params;
    for (size_t i = 0; i < params.count; i++) {
        const Node* param = params.nodes[i];
        const Type* param_type = param->payload.param.type;
        SpvId param_id = spvb_parameter(fn_builder.base, spv_emit_type(emitter, param_type));
        spv_register_emitted(emitter, false, param, param_id);
        shd_deconstruct_qualified_type(&param_type);
        if (param_type->tag == PtrType_TAG && param_type->payload.ptr_type.address_space == AsGlobal) {
            spvb_decorate(emitter->file_builder, param_id, SpvDecorationAliased, 0, NULL);
        }
    }

    if (node->payload.fun.body) {
        // reserve a bunch of identifiers for the basic blocks in the CFG
        for (size_t i = 0; i < fn_builder.cfg->size; i++) {
            CFNode* cfnode = fn_builder.cfg->rpo[i];
            assert(cfnode);
            const Node* bb = cfnode->node;
            assert(is_basic_block(bb) || bb == node);
            SpvId bb_id = spvb_fresh_id(emitter->file_builder);
            BBBuilder basic_block_builder = spvb_begin_bb(fn_builder.base, bb_id);
            shd_dict_insert(const Node*, BBBuilder, emitter->bb_builders, bb, basic_block_builder);
            // add phis for every non-entry basic block
            if (i > 0) {
                assert(is_basic_block(bb) && bb != node);
                Nodes bb_params = bb->payload.basic_block.params;
                for (size_t j = 0; j < bb_params.count; j++) {
                    const Node* bb_param = bb_params.nodes[j];
                    SpvId phi_id = spvb_fresh_id(emitter->file_builder);
                    spvb_add_phi(basic_block_builder, spv_emit_type(emitter, bb_param->type), phi_id);
                    spv_register_emitted(emitter, false, bb_param, phi_id);
                }
                // also make sure to register the label for basic blocks
                spv_register_emitted(emitter, false, bb, bb_id);
            }
        }
        emit_basic_block(emitter, &fn_builder, fn_builder.cfg->entry);

        spvb_define_function(emitter->file_builder, fn_builder.base);
    } else {
        Growy* g = shd_new_growy();
        spvb_literal_name(g, shd_get_abstraction_name(node));
        shd_growy_append_bytes(g, 4, (char*) &(uint32_t) { SpvLinkageTypeImport });
        spvb_decorate(emitter->file_builder, fn_id, SpvDecorationLinkageAttributes, shd_growy_size(g) / 4, (uint32_t*) shd_growy_data(g));
        shd_destroy_growy(g);
        spvb_declare_function(emitter->file_builder, fn_builder.base);
    }

    free(fn_builder.per_bb);
    shd_destroy_scheduler(fn_builder.scheduler);
    shd_destroy_cfg(fn_builder.cfg);
    shd_destroy_dict(fn_builder.emitted);
}

SpvId spv_emit_decl(Emitter* emitter, const Node* decl) {
    SpvId* existing = shd_dict_find_value(const Node*, SpvId, emitter->global_node_ids, decl);
    if (existing)
        return *existing;

    switch (is_declaration(decl)) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &decl->payload.global_variable;
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            spv_register_emitted(emitter, NULL, decl, given_id);
            spvb_name(emitter->file_builder, given_id, gvar->name);
            SpvId init = 0;
            if (gvar->init)
                init = spv_emit_value(emitter, NULL, gvar->init);
            SpvStorageClass storage_class = spv_emit_addr_space(emitter, gvar->address_space);
            spvb_global_variable(emitter->file_builder, given_id, spv_emit_type(emitter, decl->type), storage_class, false, init);

            for (size_t i = 0; i < decl->annotations.count; i++) {
                const Node* a = decl->annotations.nodes[i];
                assert(is_annotation(a));
                String name = get_annotation_name(a);
                if (strcmp(name, "Location") == 0) {
                    size_t loc = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(a)), false);
                    assert(loc >= 0);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationLocation, 1, (uint32_t[]) { loc });
                } else if (strcmp(name, "DescriptorSet") == 0) {
                    size_t loc = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(a)), false);
                    assert(loc >= 0);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationDescriptorSet, 1, (uint32_t[]) { loc });
                } else if (strcmp(name, "DescriptorBinding") == 0) {
                    size_t loc = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(a)), false);
                    assert(loc >= 0);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationBinding, 1, (uint32_t[]) { loc });
                }
            }

            switch (storage_class) {
                case SpvStorageClassPushConstant: {
                    break;
                }
                case SpvStorageClassStorageBuffer:
                case SpvStorageClassUniform:
                case SpvStorageClassUniformConstant: {
                    const Node* descriptor_set = shd_lookup_annotation(decl, "DescriptorSet");
                    const Node* descriptor_binding = shd_lookup_annotation(decl, "DescriptorBinding");
                    assert(descriptor_set && descriptor_binding && "DescriptorSet and/or DescriptorBinding annotations are missing");
                    break;
                }
                default: break;
            }

            shd_spv_register_interface(emitter, decl, given_id);

            return given_id;
        } case Function_TAG: {
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            spv_register_emitted(emitter, NULL, decl, given_id);
            shd_spv_emit_debuginfo(emitter, decl, given_id);
            emit_function(emitter, decl);
            return given_id;
        } case Constant_TAG: {
            // We don't emit constants at all !
            // With RefDecl, we directly grab the underlying value and emit that there and then.
            // Emitting constants as their own IDs would be nicer, but it's painful to do because decls need their ID to be reserved in advance,
            // but we also desire to cache reused values instead of emitting them multiple times. This means we can't really "force" an ID for a given value.
            // The ideal fix would be if SPIR-V offered a way to "alias" an ID under a new one. This would allow applying new debug information to the decl ID, separate from the other instances of that value.
            return 0;
        } case NominalType_TAG: {
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            spv_register_emitted(emitter, NULL, decl, given_id);
            spvb_name(emitter->file_builder, given_id, decl->payload.nom_type.name);
            spv_emit_nominal_type_body(emitter, decl->payload.nom_type.body, given_id);
            return given_id;
        }
        default: shd_error("");
    }
    shd_error("unreachable");
}

void shd_spv_emit_debuginfo(Emitter* emitter, const Node* n, SpvId id) {
    String name = shd_get_node_name_unsafe(n);
    if (name)
        spvb_name(emitter->file_builder, id, name);
}

static SpvExecutionModel emit_exec_model(ExecutionModel model) {
    switch (model) {
        case EmCompute:  return SpvExecutionModelGLCompute;
        case EmVertex:   return SpvExecutionModelVertex;
        case EmFragment: return SpvExecutionModelFragment;
        case EmNone: shd_error("No execution model but we were asked to emit it anyways");
    }
}

// First, collect all the global variables, they're needed for the interface section of OpEntryPoint
// it can be a superset of the ones actually used, so the easiest option is to just grab _all_ global variables and shove them in there
// my gut feeling says it's unlikely any drivers actually care, but validation needs to be happy so here we go...
void shd_spv_register_interface(Emitter* emitter, const Node* n, SpvId id) {
    // Prior to SPIRV 1.4, _only_ input and output variables should be found here.
    if (emitter->spirv_tgt.target_version.major == 1 &&
        emitter->spirv_tgt.target_version.minor < 4) {
        const Type* ptr_t = shd_get_unqualified_type(n->type);
        assert(ptr_t->tag == PtrType_TAG);
        switch (ptr_t->payload.ptr_type.address_space) {
            case AsOutput:
            case AsInput: break;
            default: return;
        }
    }

    shd_list_append(SpvId, emitter->interface_vars, id);
}

static void emit_entry_points(Emitter* emitter, Nodes declarations) {
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        if (decl->tag != Function_TAG) continue;
        SpvId fn_id = spv_find_emitted(emitter, NULL, decl);

        const Node* entry_point = shd_lookup_annotation(decl, "EntryPoint");
        if (entry_point) {
            ExecutionModel execution_model = shd_execution_model_from_string(shd_get_string_literal(emitter->arena, shd_get_annotation_value(entry_point)));
            assert(execution_model != EmNone);

            String exported_name = shd_get_exported_name(decl);
            assert(exported_name);
            spvb_entry_point(emitter->file_builder, emit_exec_model(execution_model), fn_id, exported_name, shd_list_count(emitter->interface_vars),shd_read_list(SpvId, emitter->interface_vars));
            emitter->num_entry_pts++;

            const Node* workgroup_size = shd_lookup_annotation(decl, "WorkgroupSize");
            if (execution_model == EmCompute)
                assert(workgroup_size);
            if (workgroup_size) {
                Nodes values = shd_get_annotation_values(workgroup_size);
                assert(values.count == 3);
                uint32_t wg_x_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[0]), false);
                uint32_t wg_y_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[1]), false);
                uint32_t wg_z_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[2]), false);

                spvb_execution_mode(emitter->file_builder, fn_id, SpvExecutionModeLocalSize, 3, (uint32_t[3]) { wg_x_dim, wg_y_dim, wg_z_dim });
            }

            if (execution_model == EmFragment) {
                spvb_execution_mode(emitter->file_builder, fn_id, SpvExecutionModeOriginUpperLeft, 0, NULL);
            }
        }
    }
}

static void emit_decls(Emitter* emitter, Nodes declarations) {
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        spv_emit_decl(emitter, decl);
    }
}

SpvId spv_get_extended_instruction_set(Emitter* emitter, const char* name) {
    SpvId* found = shd_dict_find_value(const char*, SpvId, emitter->extended_instruction_sets, name);
    if (found)
        return *found;

    SpvId new = spvb_extended_import(emitter->file_builder, name);
    shd_dict_insert(const char*, SpvId, emitter->extended_instruction_sets, name, new);
    return new;
}

RewritePass shd_spvbe_pass_map_entrypoint_args;
RewritePass shd_spvbe_pass_lift_globals_ssbo;

#include "shady/pipeline/pipeline.h"

static CompilationResult run_spv_backend_transforms(SHADY_UNUSED void* unused, const CompilerConfig* config, Module** pmod) {
    RUN_PASS(shd_pass_lower_entrypoint_args, config)
    RUN_PASS(shd_spvbe_pass_map_entrypoint_args, config)
    RUN_PASS(shd_spvbe_pass_lift_globals_ssbo, config)
    RUN_PASS(shd_pass_eliminate_constants, config)
    RUN_PASS(shd_import, config)

    return CompilationNoError;
}

void shd_pipeline_add_spirv_target_passes(ShdPipeline pipeline, SPIRVTargetConfig* econfig) {
    shd_pipeline_add_step(pipeline, (ShdPipelineStepFn) run_spv_backend_transforms, NULL, 0);
}

void shd_emit_spirv(const CompilerConfig* config, SPIRVTargetConfig target_config, Module* mod, size_t* output_size, char** output) {
    mod = shd_import(config, mod);
    IrArena* arena = shd_module_get_arena(mod);

    FileBuilder file_builder = spvb_begin();
    spvb_set_version(file_builder, target_config.target_version.major, target_config.target_version.minor);
    spvb_set_addressing_model(file_builder, SpvAddressingModelLogical);

    Emitter emitter = {
        .module = mod,
        .arena = arena,
        .configuration = config,
        .spirv_tgt = target_config,
        .file_builder = file_builder,
        .global_node_ids = shd_new_dict(Node*, SpvId, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .bb_builders = shd_new_dict(Node*, BBBuilder, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        .num_entry_pts = 0,
        .interface_vars = shd_new_list(SpvId),
    };

    emitter.extended_instruction_sets = shd_new_dict(const char*, SpvId, (HashFn) shd_hash_string, (CmpFn) shd_compare_string);

    emitter.void_t = spvb_void_type(emitter.file_builder);

    spvb_extension(file_builder, "SPV_KHR_non_semantic_info");

    Nodes decls = shd_module_get_all_exported(mod);
    emit_decls(&emitter, decls);
    emit_entry_points(&emitter, decls);

    if (emitter.num_entry_pts == 0)
        spvb_capability(file_builder, SpvCapabilityLinkage);

    spvb_capability(file_builder, SpvCapabilityShader);

    *output_size = spvb_finish(file_builder, output);

    // cleanup the emitter
    shd_destroy_dict(emitter.global_node_ids);
    shd_destroy_dict(emitter.bb_builders);
    shd_destroy_dict(emitter.extended_instruction_sets);
    shd_destroy_list(emitter.interface_vars);

    shd_destroy_ir_arena(arena);
}
