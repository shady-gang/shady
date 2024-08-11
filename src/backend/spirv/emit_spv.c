#include "emit_spv.h"

#include "shady/builtins.h"

#include "../shady/ir_private.h"
#include "../shady/analysis/cfg.h"
#include "../shady/type.h"
#include "../shady/compile.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "growy.h"
#include "util.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <analysis/scheduler.h>

extern SpvBuiltIn spv_builtins[];

#pragma GCC diagnostic error "-Wswitch"

void register_result(Emitter* emitter, bool global, const Node* node, SpvId id) {
    if (is_value(node)) {
        String name = get_value_name_unsafe(node);
        if (name)
            spvb_name(emitter->file_builder, id, name);
    }
    struct Dict* map = global ? emitter->global_node_ids : emitter->current_fn_node_ids;
    insert_dict_and_get_result(struct Node*, SpvId, map, node, id);
}

SpvId* spv_search_emitted(Emitter* emitter, const Node* node) {
    SpvId* found = find_value_dict(const Node*, SpvId, emitter->current_fn_node_ids, node);
    if (!found)
        found = find_value_dict(const Node*, SpvId, emitter->global_node_ids, node);
    return found;
}

SpvId spv_find_reserved_id(Emitter* emitter, const Node* node) {
    SpvId* found = spv_search_emitted(emitter, node);
    return *found;
}

static void emit_basic_block(Emitter* emitter, FnBuilder fn_builder, const CFG* cfg, const CFNode* cf_node) {
    const Node* bb_node = cf_node->node;
    assert(is_basic_block(bb_node) || cf_node == cfg->entry);

    const Node* body = get_abstraction_body(bb_node);

    // Find the preassigned ID to this
    BBBuilder bb_builder = spv_find_basic_block_builder(emitter, bb_node);
    SpvId bb_id = get_block_builder_id(bb_builder);
    spvb_add_bb(fn_builder, bb_builder);

    String name = get_abstraction_name_unsafe(bb_node);
    if (name)
        spvb_name(emitter->file_builder, bb_id, name);

    spv_emit_terminator(emitter, fn_builder, bb_builder, bb_node, body);
}

static void emit_function(Emitter* emitter, const Node* node) {
    assert(node->tag == Function_TAG);

    emitter->cfg = build_fn_cfg(node);
    emitter->scheduler = new_scheduler(emitter->cfg);

    const Type* fn_type = node->type;
    SpvId fn_id = spv_find_reserved_id(emitter, node);
    FnBuilder fn_builder = spvb_begin_fn(emitter->file_builder, fn_id, emit_type(emitter, fn_type), nodes_to_codom(emitter, node->payload.fun.return_types));

    Nodes params = node->payload.fun.params;
    for (size_t i = 0; i < params.count; i++) {
        const Node* param = params.nodes[i];
        const Type* param_type = param->payload.param.type;
        SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, param_type));
        register_result(emitter, false, param, param_id);
        deconstruct_qualified_type(&param_type);
        if (param_type->tag == PtrType_TAG && param_type->payload.ptr_type.address_space == AsGlobal) {
            spvb_decorate(emitter->file_builder, param_id, SpvDecorationAliased, 0, NULL);
        }
    }

    if (node->payload.fun.body) {
        CFG* cfg = build_fn_cfg(node);
        // reserve a bunch of identifiers for the basic blocks in the CFG
        for (size_t i = 0; i < cfg->size; i++) {
            CFNode* cfnode = read_list(CFNode*, cfg->contents)[i];
            assert(cfnode);
            const Node* bb = cfnode->node;
            assert(is_basic_block(bb) || bb == node);
            SpvId bb_id = spvb_fresh_id(emitter->file_builder);
            BBBuilder basic_block_builder = spvb_begin_bb(fn_builder, bb_id);
            insert_dict(const Node*, BBBuilder, emitter->bb_builders, bb, basic_block_builder);
            // if (is_cfnode_structural_target(cfnode))
            //     continue;
            // add phis for every non-entry basic block
            if (i > 0) {
                assert(is_basic_block(bb) && bb != node);
                Nodes bb_params = bb->payload.basic_block.params;
                for (size_t j = 0; j < bb_params.count; j++) {
                    const Node* bb_param = bb_params.nodes[j];
                    spvb_add_phi(basic_block_builder, emit_type(emitter, bb_param->type), spvb_fresh_id(emitter->file_builder));
                }
                // also make sure to register the label for basic blocks
                register_result(emitter, false, bb, bb_id);
            }
        }
        // emit the blocks using the dominator tree
        for (size_t i = 0; i < cfg->size; i++) {
            CFNode* cfnode = cfg->rpo[i];
            if (i == 0)
                assert(cfnode == cfg->entry);
            // if (is_cfnode_structural_target(cfnode))
            //     continue;
            emit_basic_block(emitter, fn_builder, cfg, cfnode);
        }

        destroy_cfg(cfg);

        spvb_define_function(emitter->file_builder, fn_builder);
    } else {
        Growy* g = new_growy();
        spvb_literal_name(g, get_abstraction_name(node));
        growy_append_bytes(g, 4, (char*) &(uint32_t) { SpvLinkageTypeImport });
        spvb_decorate(emitter->file_builder, fn_id, SpvDecorationLinkageAttributes, growy_size(g) / 4, (uint32_t*) growy_data(g));
        destroy_growy(g);
        spvb_declare_function(emitter->file_builder, fn_builder);
    }

    clear_dict(emitter->current_fn_node_ids);
    destroy_scheduler(emitter->scheduler);
    emitter->scheduler = NULL;
    destroy_cfg(emitter->cfg);
}

SpvId emit_decl(Emitter* emitter, const Node* decl) {
    SpvId* existing = find_value_dict(const Node*, SpvId, emitter->global_node_ids, decl);
    if (existing)
        return *existing;

    switch (is_declaration(decl)) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &decl->payload.global_variable;
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            register_result(emitter, true, decl, given_id);
            spvb_name(emitter->file_builder, given_id, gvar->name);
            SpvId init = 0;
            if (gvar->init)
                init = spv_emit_value(emitter, gvar->init);
            SpvStorageClass storage_class = emit_addr_space(emitter, gvar->address_space);
            spvb_global_variable(emitter->file_builder, given_id, emit_type(emitter, decl->type), storage_class, false, init);

            Builtin b = BuiltinsCount;
            for (size_t i = 0; i < gvar->annotations.count; i++) {
                const Node* a = gvar->annotations.nodes[i];
                assert(is_annotation(a));
                String name = get_annotation_name(a);
                if (strcmp(name, "Builtin") == 0) {
                    String builtin_name = get_annotation_string_payload(a);
                    assert(builtin_name);
                    assert(b == BuiltinsCount && "Only one @Builtin annotation permitted.");
                    b = get_builtin_by_name(builtin_name);
                    assert(b != BuiltinsCount);
                    SpvBuiltIn d = spv_builtins[b];
                    uint32_t decoration_payload[] = { d };
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationBuiltIn, 1, decoration_payload);
                } else if (strcmp(name, "Location") == 0) {
                    size_t loc = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(a)), false);
                    assert(loc >= 0);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationLocation, 1, (uint32_t[]) { loc });
                } else if (strcmp(name, "DescriptorSet") == 0) {
                    size_t loc = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(a)), false);
                    assert(loc >= 0);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationDescriptorSet, 1, (uint32_t[]) { loc });
                } else if (strcmp(name, "DescriptorBinding") == 0) {
                    size_t loc = get_int_literal_value(*resolve_to_int_literal(get_annotation_value(a)), false);
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
                    const Node* descriptor_set = lookup_annotation(decl, "DescriptorSet");
                    const Node* descriptor_binding = lookup_annotation(decl, "DescriptorBinding");
                    assert(descriptor_set && descriptor_binding && "DescriptorSet and/or DescriptorBinding annotations are missing");
                    break;
                }
                default: break;
            }

            return given_id;
        } case Function_TAG: {
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            register_result(emitter, true, decl, given_id);
            spvb_name(emitter->file_builder, given_id, decl->payload.fun.name);
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
            register_result(emitter, true, decl, given_id);
            spvb_name(emitter->file_builder, given_id, decl->payload.nom_type.name);
            spv_emit_nominal_type_body(emitter, decl->payload.nom_type.body, given_id);
            return given_id;
        }
        case NotADeclaration: error("");
    }
    error("unreachable");
}

static SpvExecutionModel emit_exec_model(ExecutionModel model) {
    switch (model) {
        case EmCompute:  return SpvExecutionModelGLCompute;
        case EmVertex:   return SpvExecutionModelVertex;
        case EmFragment: return SpvExecutionModelFragment;
        case EmNone: error("No execution model but we were asked to emit it anyways");
    }
}

static void emit_entry_points(Emitter* emitter, Nodes declarations) {
    // First, collect all the global variables, they're needed for the interface section of OpEntryPoint
    // it can be a superset of the ones actually used, so the easiest option is to just grab _all_ global variables and shove them in there
    // my gut feeling says it's unlikely any drivers actually care, but validation needs to be happy so here we go...
    LARRAY(SpvId, interface_arr, declarations.count);
    size_t interface_size = 0;
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* node = declarations.nodes[i];
        if (node->tag != GlobalVariable_TAG) continue;
        // Prior to SPIRV 1.4, _only_ input and output variables should be found here.
        if (emitter->configuration->target_spirv_version.major == 1 &&
            emitter->configuration->target_spirv_version.minor < 4) {
            switch (node->payload.global_variable.address_space) {
                case AsOutput:
                case AsInput: break;
                default: continue;
            }
        }
        interface_arr[interface_size++] = spv_find_reserved_id(emitter, node);
    }

    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        if (decl->tag != Function_TAG) continue;
        SpvId fn_id = spv_find_reserved_id(emitter, decl);

        const Node* entry_point = lookup_annotation(decl, "EntryPoint");
        if (entry_point) {
            ExecutionModel execution_model = execution_model_from_string(get_string_literal(emitter->arena, get_annotation_value(entry_point)));
            assert(execution_model != EmNone);

            spvb_entry_point(emitter->file_builder, emit_exec_model(execution_model), fn_id, decl->payload.fun.name, interface_size, interface_arr);
            emitter->num_entry_pts++;

            const Node* workgroup_size = lookup_annotation(decl, "WorkgroupSize");
            if (execution_model == EmCompute)
                assert(workgroup_size);
            if (workgroup_size) {
                Nodes values = get_annotation_values(workgroup_size);
                assert(values.count == 3);
                uint32_t wg_x_dim = (uint32_t) get_int_literal_value(*resolve_to_int_literal(values.nodes[0]), false);
                uint32_t wg_y_dim = (uint32_t) get_int_literal_value(*resolve_to_int_literal(values.nodes[1]), false);
                uint32_t wg_z_dim = (uint32_t) get_int_literal_value(*resolve_to_int_literal(values.nodes[2]), false);

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
        emit_decl(emitter, decl);
    }
}

SpvId get_extended_instruction_set(Emitter* emitter, const char* name) {
    SpvId* found = find_value_dict(const char*, SpvId, emitter->extended_instruction_sets, name);
    if (found)
        return *found;

    SpvId new = spvb_extended_import(emitter->file_builder, name);
    insert_dict(const char*, SpvId, emitter->extended_instruction_sets, name, new);
    return new;
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

static Module* run_backend_specific_passes(const CompilerConfig* config, Module* initial_mod) {
    IrArena* initial_arena = initial_mod->arena;
    Module** pmod = &initial_mod;

    RUN_PASS(lower_entrypoint_args)
    RUN_PASS(spirv_map_entrypoint_args)
    RUN_PASS(spirv_lift_globals_ssbo)
    RUN_PASS(eliminate_constants)
    RUN_PASS(import)

    return *pmod;
}

void emit_spirv(const CompilerConfig* config, Module* mod, size_t* output_size, char** output, Module** new_mod) {
    IrArena* initial_arena = get_module_arena(mod);
    mod = run_backend_specific_passes(config, mod);
    IrArena* arena = get_module_arena(mod);

    FileBuilder file_builder = spvb_begin();
    spvb_set_version(file_builder, config->target_spirv_version.major, config->target_spirv_version.minor);
    spvb_set_addressing_model(file_builder, SpvAddressingModelLogical);

    Emitter emitter = {
        .module = mod,
        .arena = arena,
        .configuration = config,
        .file_builder = file_builder,
        .global_node_ids = new_dict(Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
        .current_fn_node_ids = new_dict(Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
        .bb_builders = new_dict(Node*, BBBuilder, (HashFn) hash_node, (CmpFn) compare_node),
        .num_entry_pts = 0,
    };

    emitter.extended_instruction_sets = new_dict(const char*, SpvId, (HashFn) hash_string, (CmpFn) compare_string);

    emitter.void_t = spvb_void_type(emitter.file_builder);

    spvb_extension(file_builder, "SPV_KHR_non_semantic_info");

    Nodes decls = get_module_declarations(mod);
    emit_decls(&emitter, decls);
    emit_entry_points(&emitter, decls);

    if (emitter.num_entry_pts == 0)
        spvb_capability(file_builder, SpvCapabilityLinkage);

    spvb_capability(file_builder, SpvCapabilityShader);

    *output_size = spvb_finish(file_builder, output);

    // cleanup the emitter
    destroy_dict(emitter.global_node_ids);
    destroy_dict(emitter.current_fn_node_ids);
    destroy_dict(emitter.bb_builders);
    destroy_dict(emitter.extended_instruction_sets);

    if (new_mod)
        *new_mod = mod;
    else if (initial_arena != arena)
        destroy_ir_arena(arena);
}
