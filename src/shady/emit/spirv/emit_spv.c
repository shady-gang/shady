#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "growy.h"
#include "util.h"

#include "shady/builtins.h"
#include "../../ir_private.h"
#include "../../analysis/scope.h"
#include "../../type.h"
#include "../../compile.h"

#include "emit_spv.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>

extern SpvBuiltIn spv_builtins[];

#pragma GCC diagnostic error "-Wswitch"

void register_result(Emitter* emitter, const Node* node, SpvId id) {
    if (is_value(node)) {
        String name = get_value_name(node);
        if (name)
            spvb_name(emitter->file_builder, id, name);
    }
    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, node, id);
}

SpvId emit_value(Emitter* emitter, BBBuilder bb_builder, const Node* node) {
    SpvId* existing = find_value_dict(const Node*, SpvId, emitter->node_ids, node);
    if (existing)
        return *existing;

    SpvId new;
    switch (is_value(node)) {
        case NotAValue: error("");
        case Variable_TAG: error("tried to emit a variable: but all variables should be emitted by enclosing scope or preceding instructions !");
        case Value_ConstrainedValue_TAG:
        case Value_UntypedNumber_TAG:
        case Value_FnAddr_TAG: error("Should be lowered away earlier!");
        case IntLiteral_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            SpvId ty = emit_type(emitter, node->type);
            // 64-bit constants take two spirv words, anything else fits in one
            if (node->payload.int_literal.width == IntTy64) {
                uint32_t arr[] = { node->payload.int_literal.value & 0xFFFFFFFF, node->payload.int_literal.value >> 32 };
                spvb_constant(emitter->file_builder, new, ty, 2, arr);
            } else {
                uint32_t arr[] = { node->payload.int_literal.value };
                spvb_constant(emitter->file_builder, new, ty, 1, arr);
            }
            break;
        }
        case FloatLiteral_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            SpvId ty = emit_type(emitter, node->type);
            switch (node->payload.float_literal.width) {
                case FloatTy16: {
                    uint32_t arr[] = { node->payload.float_literal.value & 0xFFFF };
                    spvb_constant(emitter->file_builder, new, ty, 1, arr);
                    break;
                }
                case FloatTy32: {
                    uint32_t arr[] = { node->payload.float_literal.value };
                    spvb_constant(emitter->file_builder, new, ty, 1, arr);
                    break;
                }
                case FloatTy64: {
                    uint32_t arr[] = { node->payload.float_literal.value & 0xFFFFFFFF, node->payload.float_literal.value >> 32 };
                    spvb_constant(emitter->file_builder, new, ty, 2, arr);
                    break;
                }
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
        case Value_StringLiteral_TAG: {
            new = spvb_debug_string(emitter->file_builder, node->payload.string_lit.string);
            break;
        }
        case Value_NullPtr_TAG: {
            new = spvb_constant_null(emitter->file_builder, emit_type(emitter, node->payload.null_ptr.ptr_type));
            break;
        }
        case Composite_TAG: {
            Nodes contents = node->payload.composite.contents;
            LARRAY(SpvId, ids, contents.count);
            for (size_t i = 0; i < contents.count; i++) {
                ids[i] = emit_value(emitter, bb_builder, contents.nodes[i]);
            }
            if (bb_builder) {
                new = spvb_composite(bb_builder, emit_type(emitter, node->type), contents.count, ids);
                return new;
            } else {
                new = spvb_constant_composite(emitter->file_builder, emit_type(emitter, node->type), contents.count, ids);
                break;
            }
        }
        case Value_Undef_TAG: {
            new = spvb_undef(emitter->file_builder, emit_type(emitter, node->payload.undef.type));
            break;
        }
        case Value_Fill_TAG: error("lower me")
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            switch (decl->tag) {
                case GlobalVariable_TAG: {
                    new = emit_decl(emitter, decl);
                    break;
                }
                case Constant_TAG: {
                    const Node* init_value = get_quoted_value(decl->payload.constant.instruction);
                    if (!init_value && bb_builder) {
                        SpvId r;
                        emit_instruction(emitter, NULL, &bb_builder, NULL, decl->payload.constant.instruction, 1, &r);
                        return r;
                    }
                    assert(init_value && "TODO: support some measure of constant expressions");
                    new = emit_value(emitter, NULL, init_value);
                    break;
                }
                default: error("RefDecl must reference a constant or global");
            }
            break;
        }
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

static void add_branch_phis(Emitter* emitter, FnBuilder fn_builder, BBBuilder bb_builder, const Node* jump) {
    assert(jump->tag == Jump_TAG);
    const Node* dst = jump->payload.jump.target;
    Nodes args = jump->payload.jump.args;
    // because it's forbidden to jump back into the entry block of a function
    // (which is actually a Function in this IR, not a BasicBlock)
    // we assert that the destination must be an actual BasicBlock
    assert(is_basic_block(dst));
    BBBuilder dst_builder = find_basic_block_builder(emitter, fn_builder, dst);
    struct List* phis = spbv_get_phis(dst_builder);
    assert(entries_count_list(phis) == args.count);
    for (size_t i = 0; i < args.count; i++) {
        SpvbPhi* phi = read_list(SpvbPhi*, phis)[i];
        spvb_add_phi_source(phi, get_block_builder_id(bb_builder), emit_value(emitter, bb_builder, args.nodes[i]));
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
            const Node* tail = get_let_tail(terminator);
            Nodes params = tail->payload.case_.params;
            LARRAY(SpvId, results, params.count);
            emit_instruction(emitter, fn_builder, &basic_block_builder, &merge_targets, get_let_instruction(terminator), params.count, results);
            assert(tail->tag == Case_TAG);

            for (size_t i = 0; i < params.count; i++)
                register_result(emitter, params.nodes[i], results[i]);
            emit_terminator(emitter, fn_builder, basic_block_builder, merge_targets, tail->payload.case_.body);
            return;
        }
        case Jump_TAG: {
            add_branch_phis(emitter, fn_builder, basic_block_builder, terminator);
            spvb_branch(basic_block_builder, find_reserved_id(emitter, terminator->payload.jump.target));
        }
        case Branch_TAG: {
            SpvId condition = emit_value(emitter, basic_block_builder, terminator->payload.branch.branch_condition);
            add_branch_phis(emitter, fn_builder, basic_block_builder, terminator->payload.branch.true_jump);
            add_branch_phis(emitter, fn_builder, basic_block_builder, terminator->payload.branch.false_jump);
            spvb_branch_conditional(basic_block_builder, condition, find_reserved_id(emitter, terminator->payload.branch.true_jump->payload.jump.target), find_reserved_id(emitter, terminator->payload.branch.false_jump->payload.jump.target));
        }
        case Switch_TAG: {
            SpvId inspectee = emit_value(emitter, basic_block_builder, terminator->payload.br_switch.switch_value);
            LARRAY(SpvId, targets, terminator->payload.br_switch.case_jumps.count * 2);
            for (size_t i = 0; i < terminator->payload.br_switch.case_jumps.count; i++) {
                add_branch_phis(emitter, fn_builder, basic_block_builder, terminator->payload.br_switch.case_jumps.nodes[i]);
                error("TODO finish")
            }
            add_branch_phis(emitter, fn_builder, basic_block_builder, terminator->payload.br_switch.default_jump);
            SpvId default_tgt = find_reserved_id(emitter, terminator->payload.br_switch.default_jump->payload.jump.target);

            spvb_switch(basic_block_builder, inspectee, default_tgt, terminator->payload.br_switch.case_jumps.count, targets);
        }
        case LetMut_TAG:
        case TailCall_TAG:
        case Join_TAG: error("Lower me");
        case Terminator_Yield_TAG: {
            Nodes args = terminator->payload.yield.args;
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
}

static void emit_function(Emitter* emitter, const Node* node) {
    assert(node->tag == Function_TAG);

    const Type* fn_type = node->type;
    SpvId fn_id = find_reserved_id(emitter, node);
    FnBuilder fn_builder = spvb_begin_fn(emitter->file_builder, fn_id, emit_type(emitter, fn_type), nodes_to_codom(emitter, node->payload.fun.return_types));

    Nodes params = node->payload.fun.params;
    for (size_t i = 0; i < params.count; i++) {
        const Type* param_type = params.nodes[i]->payload.var.type;
        SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, param_type));
        insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, params.nodes[i], param_id);
        deconstruct_qualified_type(&param_type);
        if (param_type->tag == PtrType_TAG && param_type->payload.ptr_type.address_space == AsGlobalPhysical) {
            spvb_decorate(emitter->file_builder, param_id, SpvDecorationAliased, 0, NULL);
        }
    }

    if (node->payload.fun.body) {
        Scope* scope = new_scope(node);
        // reserve a bunch of identifiers for the basic blocks in the scope
        for (size_t i = 0; i < scope->size; i++) {
            CFNode* cfnode = read_list(CFNode*, scope->contents)[i];
            assert(cfnode);
            const Node* bb = cfnode->node;
            if (is_case(bb))
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
        for (size_t i = 0; i < scope->size; i++) {
            CFNode* cfnode = scope->rpo[i];
            if (i == 0)
                assert(cfnode == scope->entry);
            if (is_case(cfnode->node))
                continue;
            emit_basic_block(emitter, fn_builder, scope, cfnode);
        }

        destroy_scope(scope);

        spvb_define_function(emitter->file_builder, fn_builder);
    } else {
        Growy* g = new_growy();
        spvb_literal_name(g, get_abstraction_name(node));
        growy_append_bytes(g, 4, (char*) &(uint32_t) { SpvLinkageTypeImport });
        spvb_decorate(emitter->file_builder, fn_id, SpvDecorationLinkageAttributes, growy_size(g) / 4, (uint32_t*) growy_data(g));
        destroy_growy(g);
        spvb_declare_function(emitter->file_builder, fn_builder);
    }
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
            assert(!is_physical_as(gvar->address_space));
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
        interface_arr[interface_size++] = find_reserved_id(emitter, node);
    }

    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        if (decl->tag != Function_TAG) continue;
        SpvId fn_id = find_reserved_id(emitter, decl);

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

static Module* run_backend_specific_passes(CompilerConfig* config, Module* initial_mod) {
    IrArena* initial_arena = initial_mod->arena;
    Module* old_mod = NULL;
    Module** pmod = &initial_mod;

    RUN_PASS(lower_entrypoint_args)
    RUN_PASS(spirv_map_entrypoint_args)
    RUN_PASS(spirv_lift_globals_ssbo)
    RUN_PASS(eliminate_constants)
    RUN_PASS(import)

    return *pmod;
}

void emit_spirv(CompilerConfig* config, Module* mod, size_t* output_size, char** output, Module** new_mod) {
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
        .node_ids = new_dict(Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
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
    destroy_dict(emitter.node_ids);
    destroy_dict(emitter.bb_builders);
    destroy_dict(emitter.extended_instruction_sets);

    if (new_mod)
        *new_mod = mod;
    else if (initial_arena != arena)
        destroy_ir_arena(arena);
}
