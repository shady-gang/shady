#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../../ir_private.h"
#include "../../analysis/scope.h"

#include "emit_spv.h"
#include "emit_spv_builtins.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

void register_result(Emitter* emitter, const Node* variable, SpvId id) {
    spvb_name(emitter->file_builder, id, variable->payload.var.name);
    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, variable, id);
}

SpvId emit_value(Emitter* emitter, BBBuilder bb_builder, const Node* node) {
    SpvId* existing = find_value_dict(const Node*, SpvId, emitter->node_ids, node);
    if (existing)
        return *existing;

    SpvId new;
    switch (node->tag) {
        case Variable_TAG: error("tried to emit a variable: but all variables should be emitted by enclosing scope or preceding instructions !");
        case IntLiteral_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            SpvId ty = emit_type(emitter, node->type);
            // 64-bit constants take two spirv words, anythinfg else fits in one
            if (node->payload.int_literal.width == IntTy64) {
                uint32_t arr[] = { node->payload.int_literal.value.i64 >> 32, node->payload.int_literal.value.i64 & 0xFFFFFFFF };
                spvb_constant(emitter->file_builder, new, ty, 2, arr);
            } else {
                uint32_t arr[] = { node->payload.int_literal.value.i32 };
                spvb_constant(emitter->file_builder, new, ty, 1, arr);
            }
            break;
        }
        case True_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            spvb_bool_constant(emitter->file_builder, new, emit_type(emitter, bool_type(emitter->arena)), true);
            break;
        }
        case False_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            spvb_bool_constant(emitter->file_builder, new, emit_type(emitter, bool_type(emitter->arena)), false);
            break;
        }
        case Tuple_TAG: {
            Nodes components = node->payload.tuple.contents;
            LARRAY(SpvId, ids, components.count);
            for (size_t i = 0; i < components.count; i++) {
                ids[i] = emit_value(emitter, bb_builder, components.nodes[i]);
            }
            if (bb_builder) {
                new = spvb_composite(bb_builder, emit_type(emitter, node->type), components.count, ids);
                return new;
            } else {
                new = spvb_constant_composite(emitter->file_builder, emit_type(emitter, node->type), components.count, ids);
                break;
            }
        }
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            switch (decl->tag) {
                case GlobalVariable_TAG: {
                    new = *find_value_dict(const Node*, SpvId, emitter->node_ids, decl);
                    break;
                }
                case Constant_TAG: {
                    new = emit_value(emitter, NULL, decl->payload.constant.value);
                    break;
                }
                default: error("RefDecl must reference a constant or global");
            }
            break;
        }
        default: error("don't know hot to emit value");
    }

    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, node, new);
    return new;
}

SpvId spv_find_reserved_id(Emitter* emitter, const Node* node) {
    SpvId* found = find_value_dict(const Node*, SpvId, emitter->node_ids, node);
    assert(found);
    return *found;
}

static BBBuilder find_basic_block_builder(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, const Node* bb) {
    // assert(is_basic_block(bb));
    BBBuilder* found = find_value_dict(const Node*, BBBuilder, emitter->bb_builders, bb);
    assert(found);
    return *found;
}

static void add_branch_phis(Emitter* emitter, FnBuilder fn_builder, SpvId src_id, const Node* dst, size_t count, SpvId args[]) {
    // because it's forbidden to jump back into the entry block of a function
    // (which is actually a Function in this IR, not a BasicBlock)
    // we assert that the destination must be an actual BasicBlock
    assert(is_basic_block(dst));
    BBBuilder dst_builder = find_basic_block_builder(emitter, fn_builder, dst);
    struct List* phis = spbv_get_phis(dst_builder);
    assert(entries_count_list(phis) == count);
    for (size_t i = 0; i < count; i++) {
        struct Phi* phi = read_list(struct Phi*, phis)[i];
        spvb_add_phi_source(phi, src_id, args[i]);
    }
}

void emit_terminator(Emitter* emitter, FnBuilder fn_builder, BBBuilder basic_block_builder, MergeTargets merge_targets, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case Return_TAG: {
            const Nodes* ret_values = &terminator->payload.fn_ret.args;
            switch (ret_values->count) {
                case 0: spvb_return_void(basic_block_builder); return;
                case 1: spvb_return_value(basic_block_builder, emit_value(emitter, basic_block_builder, ret_values->nodes[0])); return;
                default: {
                    LARRAY(SpvId, arr, ret_values->count);
                    for (size_t i = 0; i < ret_values->count; i++)
                        arr[i] = emit_value(emitter, basic_block_builder, ret_values->nodes[i]);
                    SpvId return_that = spvb_composite(basic_block_builder, fn_ret_type_id(fn_builder), ret_values->count, arr);
                    spvb_return_value(basic_block_builder, return_that);
                    return;
                }
            }
        }
        case Let_TAG: {
            const Node* tail = terminator->payload.let.tail;
            assert(tail->tag == AnonLambda_TAG);
            Nodes params = tail->payload.anon_lam.params;
            LARRAY(SpvId, results, params.count);
            emit_instruction(emitter, fn_builder, &basic_block_builder, &merge_targets, terminator->payload.let.instruction, params.count, results);
            if (tail->tag == AnonLambda_TAG) {
                for (size_t i = 0; i < params.count; i++)
                    register_result(emitter, params.nodes[i], results[i]);
                emit_terminator(emitter, fn_builder, basic_block_builder, merge_targets, tail->payload.anon_lam.body);
            } else {
                spvb_branch(basic_block_builder, find_reserved_id(emitter, tail));
                add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), tail, params.count, results);
            }
            return;
        }
        case LetIndirect_TAG: error("TODO")
        case Jump_TAG: {
            Nodes args = terminator->payload.jump.args;
            LARRAY(SpvId, emitted_args, args.count);
            for (size_t i = 0; i < args.count; i++)
                emitted_args[i] = emit_value(emitter, basic_block_builder, args.nodes[i]);

            spvb_branch(basic_block_builder, find_reserved_id(emitter, terminator->payload.jump.target));
            add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.jump.target, args.count, emitted_args);
        }
        case Branch_TAG: {
            Nodes args = terminator->payload.branch.args;
            LARRAY(SpvId, emitted_args, args.count);
            for (size_t i = 0; i < args.count; i++)
                emitted_args[i] = emit_value(emitter, basic_block_builder, args.nodes[i]);

            SpvId condition = emit_value(emitter, basic_block_builder, terminator->payload.branch.branch_condition);
            spvb_branch_conditional(basic_block_builder, condition, find_reserved_id(emitter, terminator->payload.branch.true_target), find_reserved_id(emitter, terminator->payload.branch.false_target));

            add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.branch.true_target, args.count, emitted_args);
            add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.branch.false_target, args.count, emitted_args);
        }
        case Switch_TAG: {
            Nodes args = terminator->payload.br_switch.args;
            LARRAY(SpvId, emitted_args, args.count);
            for (size_t i = 0; i < args.count; i++)
                emitted_args[i] = emit_value(emitter, basic_block_builder, args.nodes[i]);

            error("TODO")
        }
        case LetMut_TAG:
        case TailCall_TAG:
        case Join_TAG: error("Lower me");
        case MergeSelection_TAG: {
            Nodes args = terminator->payload.merge_selection.args;
            for (size_t i = 0; i < args.count; i++)
                spvb_add_phi_source(merge_targets.join_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, basic_block_builder, args.nodes[i]));
            spvb_branch(basic_block_builder, merge_targets.join_target);
            return;
        }
        case MergeContinue_TAG: {
            Nodes args = terminator->payload.merge_continue.args;
            for (size_t i = 0; i < args.count; i++)
                spvb_add_phi_source(merge_targets.continue_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, basic_block_builder, args.nodes[i]));
            spvb_branch(basic_block_builder, merge_targets.continue_target);
            return;
        }
        case MergeBreak_TAG: {
            Nodes args = terminator->payload.merge_break.args;
            for (size_t i = 0; i < args.count; i++)
                spvb_add_phi_source(merge_targets.break_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, basic_block_builder, args.nodes[i]));
            spvb_branch(basic_block_builder, merge_targets.break_target);
            return;
        }
        case Unreachable_TAG: {
            spvb_unreachable(basic_block_builder);
            return;
        }
        case NotATerminator: error("TODO: emit terminator %s", node_tags[terminator->tag]);
    }
    SHADY_UNREACHABLE;
}

static void emit_basic_block(Emitter* emitter, FnBuilder fn_builder, const Scope* scope, const CFNode* cf_node) {
    const Node* bb_node = cf_node->node;
    assert(is_basic_block(bb_node) || cf_node == scope->entry);

    const Node* body = get_abstraction_body(bb_node);

    // Find the preassigned ID to this
    BBBuilder bb_builder = find_basic_block_builder(emitter, fn_builder, bb_node);
    SpvId bb_id = get_block_builder_id(bb_builder);
    spvb_add_bb(fn_builder, bb_builder);

    if (is_basic_block(bb_node))
        spvb_name(emitter->file_builder, bb_id, bb_node->payload.basic_block.name);

    MergeTargets merge_targets = {
        .continue_target = 0,
        .break_target = 0,
        .join_target = 0
    };

    emit_terminator(emitter, fn_builder, bb_builder, merge_targets, body);

    // Emit the child nodes
    /*size_t dom_count = entries_count_list(cf_node->dominates);
    for (size_t i = 0; i < dom_count; i++) {
        CFNode* child_node = read_list(CFNode*, cf_node->dominates)[i];
        emit_basic_block(emitter, fn_builder, scope, child_node);
    }*/
}

static void emit_function(Emitter* emitter, const Node* node) {
    assert(node->tag == Function_TAG);

    const Type* fn_type = node->type;
    FnBuilder fn_builder = spvb_begin_fn(emitter->file_builder, find_reserved_id(emitter, node), emit_type(emitter, fn_type), nodes_to_codom(emitter, node->payload.fun.return_types));

    Nodes params = node->payload.fun.params;
    for (size_t i = 0; i < params.count; i++) {
        SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, params.nodes[i]->payload.var.type));
        insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, params.nodes[i], param_id);
    }

    Scope scope = build_scope(node);
    // reserve a bunch of identifiers for the basic blocks in the scope
    for (size_t i = 0; i < scope.size; i++) {
        CFNode* cfnode = read_list(CFNode*, scope.contents)[i];
        assert(cfnode);
        const Node* bb = cfnode->node;
        if (is_anonymous_lambda(bb))
            continue;
        assert(is_basic_block(bb) || bb == node);
        SpvId bb_id = spvb_fresh_id(emitter->file_builder);
        BBBuilder basic_block_builder = spvb_begin_bb(fn_builder, bb_id);
        insert_dict(const Node*, BBBuilder, emitter->bb_builders, bb, basic_block_builder);
        // add phis for every non-entry basic block
        if (i > 0) {
            assert(is_basic_block(bb) && bb != node);
            Nodes bb_params = bb->payload.basic_block.params;
            for (size_t j = 0; j < bb_params.count; j++) {
                const Node* bb_param = bb_params.nodes[j];
                spvb_add_phi(basic_block_builder, emit_type(emitter, bb_param->type), spvb_fresh_id(emitter->file_builder));
            }
            // also make sure to register the label for basic blocks
            register_result(emitter, bb, bb_id);
        }
    }
    // emit the blocks using the dominator tree
    //emit_basic_block(emitter, fn_builder, &scope, scope.entry);
    for (size_t i = 0; i < scope.size; i++) {
        CFNode* cfnode = scope.rpo[i];
        if (i == 0)
            assert(cfnode == scope.entry);
        if (is_anonymous_lambda(cfnode->node))
            continue;
        emit_basic_block(emitter, fn_builder, &scope, cfnode);
    }

    dispose_scope(&scope);

    spvb_define_function(emitter->file_builder, fn_builder);
}

SpvId emit_decl(Emitter* emitter, const Node* decl) {
    SpvId* existing = find_value_dict(const Node*, SpvId, emitter->node_ids, decl);
    if (existing)
        return *existing;

    switch (is_declaration(decl)) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &decl->payload.global_variable;
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            register_result(emitter, decl, given_id);
            spvb_name(emitter->file_builder, given_id, gvar->name);
            SpvId init = 0;
            if (gvar->init)
                init = emit_value(emitter, NULL, gvar->init);
            SpvStorageClass storage_class = emit_addr_space(gvar->address_space);
            spvb_global_variable(emitter->file_builder, given_id, emit_type(emitter, decl->type), storage_class, false, init);

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
                    size_t set     = extract_int_literal_value(extract_annotation_value(descriptor_set),     false);
                    size_t binding = extract_int_literal_value(extract_annotation_value(descriptor_binding), false);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationDescriptorSet, 1, (uint32_t []) { set });
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationBinding, 1, (uint32_t []) { binding });
                    break;
                }
                default: break;
            }

            return given_id;
        } case Function_TAG: {
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            register_result(emitter, decl, given_id);
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
            register_result(emitter, decl, given_id);
            spvb_name(emitter->file_builder, given_id, decl->payload.nom_type.name);
            emit_nominal_type_body(emitter, decl->payload.nom_type.body, given_id);
            return given_id;
        }
        case NotADecl: error("");
    }
    error("unreachable");
}

static SpvExecutionModel emit_exec_model(ExecutionModel model) {
    switch (model) {
        case Compute:  return SpvExecutionModelGLCompute;
        case Vertex:   return SpvExecutionModelVertex;
        case Fragment: return SpvExecutionModelFragment;
        default: error("Can't emit execution model");
    }
}

static void emit_entry_points(Emitter* emitter, Nodes declarations) {
    // First, collect all the global variables, they're needed for the interface section of OpEntryPoint
    // it can be a superset of the ones actually used, so the easiest option is to just grab _all_ global variables and shove them in there
    // my gut feeling says it's unlikely any drivers actually care, but validation needs to be happy so here we go...
    LARRAY(SpvId, interface_arr, declarations.count + VulkanBuiltinsCount);
    size_t interface_size = 0;
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* node = declarations.nodes[i];
        if (node->tag != GlobalVariable_TAG) continue;
        interface_arr[interface_size++] = find_reserved_id(emitter, node);
    }
    // Do the same with builtins ...
    for (size_t i = 0; i < VulkanBuiltinsCount; i++) {
        switch (vulkan_builtins_kind[i]) {
            case VulkanBuiltinInput:
            case VulkanBuiltinOutput:
                if (emitter->emitted_builtins[i] != 0)
                    interface_arr[interface_size++] = emitter->emitted_builtins[i];
                break;
            default: error("TODO")
        }
    }

    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        if (decl->tag != Function_TAG) continue;
        SpvId fn_id = find_reserved_id(emitter, decl);

        const Node* entry_point = lookup_annotation(decl, "EntryPoint");
        if (entry_point) {
            const char* execution_model_name = extract_string_literal(extract_annotation_value(entry_point));
            SpvExecutionModel execution_model = emit_exec_model(execution_model_from_string(execution_model_name));

            spvb_entry_point(emitter->file_builder, execution_model, fn_id, decl->payload.fun.name, interface_size, interface_arr);
            emitter->num_entry_pts++;

            const Node* workgroup_size = lookup_annotation(decl, "WorkgroupSize");
            if (execution_model == SpvExecutionModelGLCompute)
                assert(workgroup_size);
            if (workgroup_size) {
                Nodes values = extract_annotation_values(workgroup_size);
                assert(values.count == 3);
                uint32_t wg_x_dim = (uint32_t) extract_int_literal_value(values.nodes[0], false);
                uint32_t wg_y_dim = (uint32_t) extract_int_literal_value(values.nodes[1], false);
                uint32_t wg_z_dim = (uint32_t) extract_int_literal_value(values.nodes[2], false);

                spvb_execution_mode(emitter->file_builder, fn_id, SpvExecutionModeLocalSize, 3, (uint32_t[3]) { wg_x_dim, wg_y_dim, wg_z_dim });
            }
        }
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static void emit_decls(Emitter* emitter, Nodes declarations) {
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        emit_decl(emitter, decl);
    }
}

void emit_spirv(CompilerConfig* config, Module* mod, size_t* output_size, char** output) {
    IrArena* arena = get_module_arena(mod);
    struct List* words = new_list(uint32_t);

    FileBuilder file_builder = spvb_begin();
    spvb_set_version(file_builder, config->target_spirv_version.major, config->target_spirv_version.minor);

    Emitter emitter = {
        .module = mod,
        .arena = arena,
        .configuration = config,
        .file_builder = file_builder,
        .node_ids = new_dict(Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
        .bb_builders = new_dict(Node*, BBBuilder, (HashFn) hash_node, (CmpFn) compare_node),
        .num_entry_pts = 0,
    };

    emitter.non_semantic_imported_instrs.debug_printf = spvb_extended_import(file_builder, "NonSemantic.DebugPrintf");

    for (size_t i = 0; i < VulkanBuiltinsCount; i++)
        emitter.emitted_builtins[i] = 0;

    emitter.void_t = spvb_void_type(emitter.file_builder);

    spvb_extension(file_builder, "SPV_KHR_non_semantic_info");
    spvb_extension(file_builder, "SPV_KHR_physical_storage_buffer");

    Nodes decls = get_module_declarations(mod);
    emit_decls(&emitter, decls);
    emit_entry_points(&emitter, decls);

    if (emitter.num_entry_pts == 0)
        spvb_capability(file_builder, SpvCapabilityLinkage);

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityInt64);
    spvb_capability(file_builder, SpvCapabilityPhysicalStorageBufferAddresses);
    spvb_capability(file_builder, SpvCapabilityGroupNonUniform);
    spvb_capability(file_builder, SpvCapabilityGroupNonUniformBallot);

    spvb_finish(file_builder, words);

    // cleanup the emitter
    destroy_dict(emitter.node_ids);
    destroy_dict(emitter.bb_builders);

    *output_size = words->elements_count * sizeof(uint32_t);
    *output = malloc(*output_size);
    memcpy(*output, words->alloc, *output_size);

    destroy_list(words);
}
