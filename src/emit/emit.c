#include "list.h"
#include "dict.h"

#include "../log.h"
#include "../local_array.h"
#include "../type.h"
#include "../analysis/scope.h"

#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct Emitter_ {
    CompilerConfig* configuration;
    IrArena* arena;
    struct SpvFileBuilder* file_builder;
    SpvId void_t;
    struct Dict* node_ids;
} Emitter;

typedef struct BBEmissionCtx_ {
    struct SpvBasicBlockBuilder* basic_block_builder;
    SpvId reserved_id;
    int case_id;
} BBEmissionCtx;

typedef struct FunctionEmissionCtx_ {
    struct SpvFnBuilder* fn_builder;
    // basic blocks use a different namespace because the entry of a function is the function itself
    // in other words, the function node gets _two_ ids: one emitted as a function, and another as the entry BB
    struct Dict* bbs;
    SpvId fn_loop_continue;
    SpvId next_bb_var;
} FunctionEmissionCtx;

const int AsSpecialFn = 0x100;

SpvStorageClass emit_addr_space(AddressSpace address_space) {
    switch(address_space) {
        case AsGeneric: return SpvStorageClassGeneric;
        case AsPrivate: return SpvStorageClassPrivate;
        case AsShared: return SpvStorageClassCrossWorkgroup;
        case AsInput: return SpvStorageClassInput;
        case AsOutput: return SpvStorageClassOutput;
        case AsGlobal: return SpvStorageClassPhysicalStorageBuffer;

        // TODO: depending on platform, use push constants/ubos/ssbos here
        case AsExternal: return SpvStorageClassStorageBuffer;

        case 0x100: return SpvStorageClassFunction;
        default: SHADY_NOT_IMPLEM;
    }
}

SpvId emit_type(Emitter* emitter, const Type* type);
SpvId emit_value(Emitter* emitter, const Node* node, const SpvId* use_id);

void emit_primop(Emitter* emitter, struct SpvFnBuilder* fnb, struct SpvBasicBlockBuilder* bbb, Op op, Nodes args, SpvId out[]) {
    LARRAY(SpvId, arr, args.count);
    for (size_t i = 0; i < args.count; i++)
        arr[i] = emit_value(emitter, args.nodes[i], NULL);

    SpvId i32_t = emit_type(emitter, int_type(emitter->arena));

    switch (op) {
        case add_op: out[0] = spvb_binop(bbb, SpvOpIAdd, i32_t, arr[0], arr[1]); break;
        case sub_op: out[0] = spvb_binop(bbb, SpvOpISub, i32_t, arr[0], arr[1]); break;
        default: error("TODO: unhandled op");
    }
}

struct SpvBasicBlockBuilder* emit_instruction(Emitter* emitter, struct SpvFnBuilder* fnb, struct SpvBasicBlockBuilder* bbb, const Node* instruction) {
    switch (instruction->tag) {
        case Let_TAG: {
            const Nodes* variables = &instruction->payload.let.variables;
            LARRAY(SpvId, out, variables->count);
            emit_primop(emitter, fnb, bbb, instruction->payload.let.op, instruction->payload.let.args, out);
            for (size_t i = 0; i < variables->count; i++) {
                spvb_name(emitter->file_builder, out[i], variables->nodes[i]->payload.var.name);
                insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, variables->nodes[i], out[i]);
            }
            return bbb;
        }
        case IfInstr_TAG: error("we expect this stuff to be gotten rid of by now actually")
        default: error("TODO: emit instruction");
    }
    SHADY_UNREACHABLE;
}

BBEmissionCtx* find_bb_ctx(FunctionEmissionCtx* fn_ectx, const Node* bb) {
    BBEmissionCtx* found = find_value_dict(struct Node*, BBEmissionCtx, fn_ectx->bbs, bb);
    assert(found);
    return found;
}

void emit_terminator(Emitter* emitter, FunctionEmissionCtx* fn_ectx, BBEmissionCtx* bb_ectx, const Node* terminator) {
    switch (terminator->tag) {
        case Return_TAG: {
            const Nodes* ret_values = &terminator->payload.fn_ret.values;
            switch (ret_values->count) {
                case 0: spvb_return_void(bb_ectx->basic_block_builder); return;
                case 1: spvb_return_value(bb_ectx->basic_block_builder, emit_value(emitter, ret_values->nodes[0], NULL)); return;
                default: {
                    LARRAY(SpvId, arr, ret_values->count);
                    for (size_t i = 0; i < ret_values->count; i++)
                        arr[i] = emit_value(emitter, ret_values->nodes[i], NULL);
                    SpvId return_that = spvb_composite(bb_ectx->basic_block_builder, fn_ret_type_id(fn_ectx->fn_builder), ret_values->count, arr);
                    spvb_return_value(bb_ectx->basic_block_builder, return_that);
                    return;
                }
            }
        }
        case Jump_TAG: {
            assert(terminator->payload.jump.args.count == 0 && "TODO: implement bb params");

            BBEmissionCtx* tgt_ctx = find_bb_ctx(fn_ectx, terminator->payload.jump.target);
            if (emitter->configuration->use_loop_for_fn_body) {
                SpvId tgt_case_id = emit_value(emitter, int_literal(emitter->arena, (IntLiteral) {
                    .value = tgt_ctx->case_id,
                }), NULL);
                spvb_store(bb_ectx->basic_block_builder, tgt_case_id, fn_ectx->next_bb_var, 0, NULL);
                spvb_branch(bb_ectx->basic_block_builder, fn_ectx->fn_loop_continue);
            } else {
                spvb_branch(bb_ectx->basic_block_builder, tgt_ctx->reserved_id);
            }
            return;
        }
        case Branch_TAG: {
            assert(terminator->payload.branch.args.count == 0 && "TODO: implement bb params");
            BBEmissionCtx* if_true_ctx = find_bb_ctx(fn_ectx, terminator->payload.branch.true_target);
            BBEmissionCtx* if_false_ctx = find_bb_ctx(fn_ectx, terminator->payload.branch.false_target);

            SpvId condition = emit_value(emitter, terminator->payload.branch.condition, NULL);

            if (emitter->configuration->use_loop_for_fn_body) {
                SpvId true_tgt_case_id = emit_value(emitter, int_literal(emitter->arena, (IntLiteral) {
                    .value = if_true_ctx->case_id,
                }), NULL);
                SpvId false_tgt_case_id = emit_value(emitter, int_literal(emitter->arena, (IntLiteral) {
                    .value = if_false_ctx->case_id,
                }), NULL);
                SpvId tgt_case_id = spvb_select(bb_ectx->basic_block_builder, emit_type(emitter, bool_type(emitter->arena)), condition, true_tgt_case_id, false_tgt_case_id);
                spvb_store(bb_ectx->basic_block_builder, tgt_case_id, fn_ectx->next_bb_var, 0, NULL);
                spvb_branch(bb_ectx->basic_block_builder, fn_ectx->fn_loop_continue);
            } else {
                spvb_branch_conditional(bb_ectx->basic_block_builder, condition, if_true_ctx->reserved_id, if_false_ctx->reserved_id);
            }
            return;
        }
        case Join_TAG: error("the join terminator is supposed to be eliminated in the instr2bb pass");
        case Unreachable_TAG: {
            spvb_unreachable(bb_ectx->basic_block_builder);
            return;
        }
        default: error("TODO: emit terminator %s", node_tags[terminator->tag]);
    }
    SHADY_UNREACHABLE;
}

void emit_basic_block(Emitter* emitter, FunctionEmissionCtx* fn_ectx, const CFNode* node) {
    // Find the preassigned ID to this
    BBEmissionCtx* bb_ectx = find_bb_ctx(fn_ectx, node->node);
    bb_ectx->basic_block_builder = spvb_begin_bb(fn_ectx->fn_builder, bb_ectx->reserved_id);
    spvb_name(emitter->file_builder, bb_ectx->reserved_id, node->node->payload.fn.name);

    struct SpvBasicBlockBuilder* basicblock_builder = bb_ectx->basic_block_builder;
    const Block* block = &node->node->payload.fn.block->payload.block;
    for (size_t i = 0; i < block->instructions.count; i++)
        basicblock_builder = emit_instruction(emitter, fn_ectx->fn_builder, basicblock_builder, block->instructions.nodes[i]);
    emit_terminator(emitter, fn_ectx, bb_ectx, block->terminator);

    // Emit the child nodes for real
    size_t dom_count = entries_count_list(node->dominates);
    for (size_t i = 0; i < dom_count; i++) {
        CFNode* child_node = read_list(CFNode*, node->dominates)[i];
        emit_basic_block(emitter, fn_ectx, child_node);
    }
}

static SpvId nodes_to_codom(Emitter* emitter, Nodes return_types) {
    switch (return_types.count) {
        case 0: return emitter->void_t;
        case 1: return emit_type(emitter, return_types.nodes[0]);
        default: {
            const Type* codom_ret_type = record_type(emitter->arena, (RecordType) {.members = return_types});
            return emit_type(emitter, codom_ret_type);
        }
    }
}

SpvId emit_function(Emitter* emitter, const Node* node, const SpvId new) {
    assert(node->tag == Function_TAG);
    const Node* fn_type = derive_fn_type(emitter->arena, &node->payload.fn);
    struct SpvFnBuilder* fn_builder = spvb_begin_fn(emitter->file_builder, new, emit_type(emitter, fn_type), nodes_to_codom(emitter, node->payload.fn.return_types));
    Nodes params = node->payload.fn.params;
    for (size_t i = 0; i < params.count; i++) {
        SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, params.nodes[i]->payload.var.type));
        insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, params.nodes[i], param_id);
    }

    FunctionEmissionCtx fn_emit_ctx = {
        .bbs = new_dict(struct Node*, BBEmissionCtx, (HashFn) hash_node, (CmpFn) compare_node),
        .fn_builder = fn_builder,
        .next_bb_var = -1,
        .fn_loop_continue = -1,
    };

    Scope scope = build_scope(node);

    // Reserve and assign IDs for basic blocks within this
    int case_id = 0;
    for (size_t i = 0; i < scope.size; i++) {
        CFNode* bb = read_list(CFNode*, scope.contents)[i];
        SpvId reserved_id = spvb_fresh_id(emitter->file_builder);
        BBEmissionCtx bb_ectx = {
            .reserved_id = reserved_id,
            .case_id = case_id++,
            .basic_block_builder = NULL,
        };
        insert_dict_and_get_result(struct Node*, BBEmissionCtx, fn_emit_ctx.bbs, bb->node, bb_ectx);
    }

    SpvId int_spv = emit_type(emitter, int_type(emitter->arena));
    SpvId ptr_int_spv = emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
        .address_space = AsSpecialFn,
        .pointed_type = int_type(emitter->arena)
    }));

    SpvId loop_unreachable_bb = spvb_fresh_id(emitter->file_builder);
        spvb_name(emitter->file_builder, loop_unreachable_bb, "loop_unreachable");

    if (emitter->configuration->use_loop_for_fn_body) {
        fn_emit_ctx.next_bb_var = spvb_local_variable(fn_builder, ptr_int_spv, SpvStorageClassFunction);
        fn_emit_ctx.fn_loop_continue = spvb_fresh_id(emitter->file_builder);
        SpvId entry_block_id = spvb_fresh_id(emitter->file_builder);
        SpvId loop_header_id = spvb_fresh_id(emitter->file_builder);
        SpvId loop_body_id = spvb_fresh_id(emitter->file_builder);
        SpvId loop_dispatch_merge_id = spvb_fresh_id(emitter->file_builder);

        spvb_name(emitter->file_builder, entry_block_id, "entry_block");
        spvb_name(emitter->file_builder, loop_header_id, "loop_header");
        spvb_name(emitter->file_builder, loop_body_id, "loop_body");
        spvb_name(emitter->file_builder, loop_dispatch_merge_id, "loop_dispatch_merge");
        spvb_name(emitter->file_builder, fn_emit_ctx.fn_loop_continue, "loop_continue");

        struct SpvBasicBlockBuilder* fn_entry = spvb_begin_bb(fn_builder, entry_block_id);
        spvb_branch(fn_entry, loop_header_id);

        struct SpvBasicBlockBuilder* loop_header = spvb_begin_bb(fn_builder, loop_header_id);
        spvb_loop_merge(loop_header, loop_unreachable_bb, fn_emit_ctx.fn_loop_continue, SpvLoopControlDontUnrollMask, 0, NULL);
        spvb_branch(loop_header, loop_body_id);

        struct SpvBasicBlockBuilder* loop_body = spvb_begin_bb(fn_builder, loop_body_id);
        LARRAY(SpvId, targets, scope.size * 2);
        for (size_t i = 0; i < scope.size; i++) {
            CFNode* bb = read_list(CFNode*, scope.contents)[i];
            BBEmissionCtx* bb_ectx = find_bb_ctx(&fn_emit_ctx, bb->node);
            targets[i * 2 + 0] = bb_ectx->case_id;
            targets[i * 2 + 1] = bb_ectx->reserved_id;
        }
        SpvId next_bb = spvb_load(loop_body, int_spv, fn_emit_ctx.next_bb_var, 0, NULL);
        spvb_selection_merge(loop_body, loop_dispatch_merge_id, 0);
        spvb_switch(loop_body, next_bb, loop_dispatch_merge_id /* todo unreachable ? */, scope.size, targets);

        struct SpvBasicBlockBuilder* loop_dispatch_merge_bb = spvb_begin_bb(fn_builder, loop_dispatch_merge_id);
        spvb_branch(loop_dispatch_merge_bb, fn_emit_ctx.fn_loop_continue);

        struct SpvBasicBlockBuilder* unreachable_bb = spvb_begin_bb(fn_builder, loop_unreachable_bb);
        spvb_unreachable(unreachable_bb);
        spvb_name(emitter->file_builder, loop_unreachable_bb, "dummy unreachable for the function BB loop");

        struct SpvBasicBlockBuilder* loop_continue_bb = spvb_begin_bb(fn_builder, fn_emit_ctx.fn_loop_continue);
        spvb_branch(loop_continue_bb, loop_header_id);
    }
    emit_basic_block(emitter, &fn_emit_ctx, scope.entry);
    dispose_scope(&scope);

    destroy_dict(fn_emit_ctx.bbs);

    spvb_define_function(emitter->file_builder, fn_builder);
}

SpvId emit_value(Emitter* emitter, const Node* node, const SpvId* use_id) {
    SpvId* existing = find_value_dict(struct Node*, SpvId, emitter->node_ids, node);
    if (existing)
        return *existing;

    SpvId new = use_id ? *use_id : spvb_fresh_id(emitter->file_builder);
    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, node, new);

    switch (node->tag) {
        case Variable_TAG: error("this variable should have been resolved already");
        case IntLiteral_TAG: {
            SpvId ty = emit_type(emitter, node->type);
            uint32_t arr[] = { node->payload.int_literal.value };
            spvb_constant(emitter->file_builder, new, ty, 1, arr);
            break;
        }
        case True_TAG: {
            spvb_bool_constant(emitter->file_builder, new, emit_type(emitter, bool_type(emitter->arena)), true);
            break;
        }
        case False_TAG: {
            spvb_bool_constant(emitter->file_builder, new, emit_type(emitter, bool_type(emitter->arena)), false);
            break;
        }
        default: error("don't know hot to emit value");
    }
    return new;
}

SpvId emit_type(Emitter* emitter, const Type* type) {
    SpvId* existing = find_value_dict(struct Node*, SpvId, emitter->node_ids, type);
    if (existing)
        return *existing;
    
    SpvId new;
    switch (type->tag) {
        case Int_TAG:
            new = spvb_int_type(emitter->file_builder, 32, true);
            break;
        case Bool_TAG:
            new = spvb_bool_type(emitter->file_builder);
            break;
        case PtrType_TAG: {
            SpvId pointee = emit_type(emitter, type->payload.ptr_type.pointed_type);
            SpvStorageClass sc = emit_addr_space(type->payload.ptr_type.address_space);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);
            break;
        }
        case RecordType_TAG: {
            LARRAY(SpvId, members, type->payload.record_type.members.count);
            for (size_t i = 0; i < type->payload.record_type.members.count; i++)
                members[i] = emit_type(emitter, type->payload.record_type.members.nodes[i]);
            new = spvb_struct_type(emitter->file_builder, type->payload.record_type.members.count, members);
            break;
        }
        case FnType_TAG: {
            const FnType* fnt = &type->payload.fn_type;
            assert(!fnt->is_continuation);
            LARRAY(SpvId, params, fnt->param_types.count);
            for (size_t i = 0; i < fnt->param_types.count; i++)
                params[i] = emit_type(emitter, fnt->param_types.nodes[i]);

            new = spvb_fn_type(emitter->file_builder, fnt->param_types.count, params, nodes_to_codom(emitter, fnt->return_types));
            break;
        }
        case QualifiedType_TAG: {
            // SPIR-V does not care about our type qualifiers.
            new = emit_type(emitter, type->payload.qualified_type.type);
            break;
        }
        default: error("Don't know how to emit type")
    }

    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, type, new);
    return new;
}

void emit_spirv(CompilerConfig* config, IrArena* arena, const Node* root_node, FILE* output) {
    const Root* top_level = &root_node->payload.root;
    struct List* words = new_list(uint32_t);

    struct SpvFileBuilder* file_builder = spvb_begin();

    Emitter emitter = {
        .configuration = config,
        .arena = arena,
        .file_builder = file_builder,
        .node_ids = new_dict(struct Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
    };

    emitter.void_t = spvb_void_type(emitter.file_builder);

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityLinkage);
    spvb_capability(file_builder, SpvCapabilityPhysicalStorageBufferAddresses);

    LARRAY(SpvId, ids, top_level->variables.count);
    for (size_t i = 0; i < top_level->variables.count; i++) {
        const Node* variable = top_level->variables.nodes[i];
        ids[i] = spvb_fresh_id(file_builder);
        insert_dict_and_get_result(struct Node*, SpvId, emitter.node_ids, variable, ids[i]);
        spvb_name(file_builder, ids[i], variable->payload.var.name);
    }

    for (size_t i = 0; i < top_level->variables.count; i++) {
        const Node* definition = top_level->definitions.nodes[i];

        DivergenceQualifier qual;
        const Type* type = strip_qualifier(top_level->variables.nodes[i]->payload.var.type, &qual);

        if (definition == NULL) {
            assert(qual == Uniform && "the _pointers_ to externals (descriptors mostly) should be uniform");
            assert(type->tag == PtrType_TAG);
            spvb_global_variable(file_builder, ids[i], emit_type(&emitter, type), emit_addr_space(type->payload.ptr_type.address_space), false, 0);
            continue;
        }

        definition->tag == Function_TAG ? emit_function(&emitter, definition, ids[i]) : emit_value(&emitter, definition, &ids[i]);
    }

    spvb_finish(file_builder, words);
    destroy_dict(emitter.node_ids);

    fwrite(words->alloc, words->elements_count, 4, output);

    destroy_list(words);
}