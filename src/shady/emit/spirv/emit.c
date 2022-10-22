#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../../type.h"
#include "../../analysis/scope.h"

#include "emit.h"
#include "emit_builtins.h"
#include "emit_type.h"

#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "spirv/unified1/NonSemanticDebugPrintf.h"

#pragma GCC diagnostic error "-Wswitch"

typedef struct Phi** Phis;

typedef struct {
    SpvId continue_target, break_target, join_target;
    Phis continue_phis, break_phis, join_phis;
} MergeTargets;

// static void emit_body(Emitter* emitter, FnBuilder fn_builder, BBBuilder basic_block_builder, MergeTargets merge_targets, const Node* node);

static void register_result(Emitter* emitter, const Node* variable, SpvId id) {
    spvb_name(emitter->file_builder, id, variable->payload.var.name);
    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, variable, id);
}

SpvId emit_value(Emitter* emitter, const Node* node) {
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
                uint32_t arr[] = { node->payload.int_literal.value_i64 >> 32, node->payload.int_literal.value_i64 & 0xFFFFFFFF };
                spvb_constant(emitter->file_builder, new, ty, 2, arr);
            } else {
                uint32_t arr[] = { node->payload.int_literal.value_i32 };
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
        case RefDecl_TAG: {
            const Node* decl = node->payload.ref_decl.decl;
            switch (decl->tag) {
                case GlobalVariable_TAG: {
                    new = *find_value_dict(const Node*, SpvId, emitter->node_ids, decl);
                    break;
                }
                case Constant_TAG: {
                    new = emit_value(emitter, decl->payload.constant.value);
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

enum OperandKind {
    Signed, Unsigned, Float, Logical, Ptr, Other, OperandKindsCount
};

enum ResultKind {
    Same, Bool, TyOperand
};

static enum OperandKind classify_primop_arg(const Node* arg) {
    const Type* operand_type = is_type(arg) ? arg : extract_operand_type(arg->type);
    assert(!contains_qualified_type(operand_type));

    switch (operand_type->tag) {
        case Int_TAG:     return Signed;
        case Bool_TAG:    return Logical;
        case PtrType_TAG: return Ptr;
        // TODO float, unsigned
        default: error("we don't know what to do with this")
    }
}

/// What is considered when searching for an instruction
enum ISelMechanism {
    Custom, FirstOp, FirstAndResult
};

#define ISEL_IDENTITY (SpvOpNop /* no-op, should be lowered to nothing beforehand */)
#define ISEL_LOWERME (SpvOpMax /* boolean conversions don't exist as a single instruction, a pass should lower them instead */)
#define ISEL_ILLEGAL (SpvOpMax /* doesn't make sense to support */)
#define ISEL_CUSTOM (SpvOpMax /* doesn't make sense to support */)

const struct IselTableEntry {
    enum ISelMechanism i_sel_mechanism;
    enum ResultKind result_kind;
    union {
        // matches first operand
        SpvOp fo[OperandKindsCount];
        // matches first operand and return type [first operand][result type]
        SpvOp foar[OperandKindsCount][OperandKindsCount];
    };
} isel_table[] = {
    [add_op] = { FirstOp, Same, .fo = { SpvOpIAdd, SpvOpIAdd, SpvOpFAdd }},
    [sub_op] = { FirstOp, Same, .fo = { SpvOpISub, SpvOpISub, SpvOpFSub }},
    [mul_op] = { FirstOp, Same, .fo = { SpvOpIMul, SpvOpIMul, SpvOpFMul }},
    [div_op] = { FirstOp, Same, .fo = { SpvOpSDiv, SpvOpUDiv, SpvOpFDiv }},
    [mod_op] = { FirstOp, Same, .fo = { SpvOpSMod, SpvOpUMod, SpvOpFMod }},

    [neg_op] = { FirstOp, Same, .fo = { SpvOpSNegate, SpvOpSNegate }},

    [eq_op]  = { FirstOp, Bool, .fo = { SpvOpIEqual,            SpvOpIEqual,            SpvOpFOrdNotEqual,         SpvOpLogicalEqual    }},
    [neq_op] = { FirstOp, Bool, .fo = { SpvOpINotEqual,         SpvOpINotEqual,         SpvOpFOrdNotEqual,         SpvOpLogicalNotEqual }},
    [lt_op]  = { FirstOp, Bool, .fo = { SpvOpSLessThan,         SpvOpULessThan,         SpvOpFOrdLessThan,         ISEL_IDENTITY        }},
    [lte_op] = { FirstOp, Bool, .fo = { SpvOpSLessThanEqual,    SpvOpULessThanEqual,    SpvOpFOrdLessThanEqual,    ISEL_IDENTITY        }},
    [gt_op]  = { FirstOp, Bool, .fo = { SpvOpSGreaterThan,      SpvOpUGreaterThan,      SpvOpFOrdGreaterThan,      ISEL_IDENTITY        }},
    [gte_op] = { FirstOp, Bool, .fo = { SpvOpSGreaterThanEqual, SpvOpUGreaterThanEqual, SpvOpFOrdGreaterThanEqual, ISEL_IDENTITY        }},

    [not_op] = { FirstOp, Same, .fo = { SpvOpNot,        SpvOpNot,        SpvOpLogicalNot      }},
    [and_op] = { FirstOp, Same, .fo = { SpvOpBitwiseAnd, SpvOpBitwiseAnd, SpvOpLogicalAnd      }},
    [or_op]  = { FirstOp, Same, .fo = { SpvOpBitwiseOr,  SpvOpBitwiseOr,  SpvOpLogicalOr       }},
    [xor_op] = { FirstOp, Same, .fo = { SpvOpBitwiseXor, SpvOpBitwiseXor, SpvOpLogicalNotEqual }},

    [lshift_op]         = { FirstOp, Same, .fo = { SpvOpShiftLeftLogical,     SpvOpShiftLeftLogical,     ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_arithm_op]  = { FirstOp, Same, .fo = { SpvOpShiftRightArithmetic, SpvOpShiftRightArithmetic, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_logical_op] = { FirstOp, Same, .fo = { SpvOpShiftRightLogical,    SpvOpShiftRightLogical,    ISEL_ILLEGAL, ISEL_ILLEGAL }},

    [convert_op] = { FirstAndResult, TyOperand, .foar = {
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertSToF, ISEL_LOWERME  },
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertUToF, ISEL_LOWERME  },
        { SpvOpConvertFToS, SpvOpConvertFToU, SpvOpFConvert,    ISEL_ILLEGAL  },
        { ISEL_LOWERME,     ISEL_LOWERME,     ISEL_ILLEGAL,     ISEL_IDENTITY }
    }},

    [reinterpret_op] = { FirstAndResult, TyOperand, .foar = {
        { SpvOpUConvert,      SpvOpBitcast,       SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       ISEL_IDENTITY,      SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       SpvOpBitcast,       ISEL_IDENTITY, ISEL_ILLEGAL,  ISEL_ILLEGAL /* no fp-ptr casts */ },
        { ISEL_ILLEGAL,       ISEL_ILLEGAL,       ISEL_ILLEGAL,  ISEL_IDENTITY, ISEL_ILLEGAL /* no bool reinterpret */ },
        { SpvOpConvertPtrToU, SpvOpConvertPtrToU, ISEL_ILLEGAL,  ISEL_ILLEGAL,  ISEL_IDENTITY }
    }},

    [PRIMOPS_COUNT] = { .i_sel_mechanism = Custom }
};

static void emit_primop(Emitter* emitter, FnBuilder fn_builder, BBBuilder bb_builder, const Node* instr, size_t results_count, SpvId results[]) {
    PrimOp prim_op = instr->payload.prim_op;
    Nodes args = prim_op.operands;

    struct IselTableEntry entry = isel_table[prim_op.op];
    if (entry.i_sel_mechanism != Custom) {
        LARRAY(SpvId, arr, args.count);
        for (size_t i = 0; i < args.count; i++) {
            if (!args.nodes[i])
                continue;
            else if (is_type(args.nodes[i]))
                arr[i] = emit_type(emitter, args.nodes[i]);
            else
                arr[i] = emit_value(emitter, args.nodes[i]);
        }

        SpvOp opcode;
        int first_op = 0;
        if (entry.i_sel_mechanism == FirstOp) {
            enum OperandKind op_class = classify_primop_arg(args.nodes[0]);
            opcode = entry.fo[op_class];
        } else if (entry.i_sel_mechanism == FirstAndResult) {
            enum OperandKind return_t_class = classify_primop_arg(args.nodes[0]);
            enum OperandKind op_class = classify_primop_arg(args.nodes[1]);
            opcode = entry.foar[op_class][return_t_class];
            first_op = 1;
        } else SHADY_UNREACHABLE;

        if (opcode == SpvOpNop) {
            assert(variables.count == 1);
            register_result(emitter, variables.nodes[0], arr[first_op]);
            return;
        } else if (opcode == SpvOpMax) {
            goto custom_path;
        }

        const Type* result_t;
        switch (entry.result_kind) {
            case Same:      result_t = extract_operand_type(args.nodes[0]->type); break;
            case Bool:      result_t = bool_type(emitter->arena); break;
            case TyOperand: result_t = args.nodes[0]; break;
            default: error("unhandled result kind");
        }

        if (args.count == 1 || first_op == 1)
            register_result(emitter, variables.nodes[0], spvb_unop(bb_builder, opcode, emit_type(emitter, result_t), arr[first_op]));
        else if (args.count == 2)
            register_result(emitter, variables.nodes[0], spvb_binop(bb_builder, opcode, emit_type(emitter, result_t), arr[0], arr[1]));
        else
            error("unhandled isel for argsc > 2");

        return;
    }

    custom_path:
    switch (prim_op.op) {
        case subgroup_ballot_op: {
            const Type* i32x4 = pack_type(emitter->arena, (PackType) { .width = 4, .element_type = int32_type(emitter->arena) });
            SpvId scope_subgroup = emit_value(emitter, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_ballot(bb_builder, emit_type(emitter, i32x4), emit_value(emitter, args.nodes[0]), scope_subgroup);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case subgroup_broadcast_first_op: {
            SpvId scope_subgroup = emit_value(emitter, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_broadcast_first(bb_builder, emit_type(emitter, extract_operand_type(args.nodes[0]->type)), emit_value(emitter, args.nodes[0]), scope_subgroup);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case subgroup_local_id_op: {
            SpvId result_t = emit_type(emitter, get_vulkan_builtins_type(emitter->arena, VulkanBuiltinSubgroupLocalInvocationId));
            SpvId ptr = emit_builtin(emitter, VulkanBuiltinSubgroupLocalInvocationId);
            SpvId result = spvb_load(bb_builder, result_t, ptr, 0, NULL);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case subgroup_elect_first_op: {
            SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
            SpvId scope_subgroup = emit_value(emitter, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_elect(bb_builder, result_t, scope_subgroup);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case extract_op: {
            const Node* src_value = args.nodes[0];
            const Type* result_t = instr->type;
            LARRAY(uint32_t, arr, args.count - 1);
            for (size_t i = 0; i < args.count - 1; i++) {
                arr[i] = extract_int_literal_value(args.nodes[i + 1], false);
            }
            assert(args.count > 1);
            SpvId result = spvb_extract(bb_builder, emit_type(emitter, result_t), emit_value(emitter, src_value), args.count - 1, arr);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        /*case reinterpret_op: {
            const Type* dst_type = args.nodes[0];
            assert(dst_type->tag == Int_TAG);
            const Type* src_type = extract_operand_type(args.nodes[1]->type);
            assert(src_type->tag == Int_TAG);
        }*/
        case load_op: {
            assert(extract_operand_type(args.nodes[0]->type)->tag == PtrType_TAG);
            const Type* elem_type = extract_operand_type(args.nodes[0]->type)->payload.ptr_type.pointed_type;
            SpvId eptr = emit_value(emitter, args.nodes[0]);
            SpvId result = spvb_load(bb_builder, emit_type(emitter, elem_type), eptr, 0, NULL);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case store_op: {
            assert(extract_operand_type(args.nodes[0]->type)->tag == PtrType_TAG);
            SpvId eptr = emit_value(emitter, args.nodes[0]);
            SpvId eval = emit_value(emitter, args.nodes[1]);
            spvb_store(bb_builder, eval, eptr, 0, NULL);
            return;
        }
        case alloca_logical_op: {
            const Type* elem_type = args.nodes[0];
            SpvId result = spvb_local_variable(fn_builder, emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunctionLogical,
                .pointed_type = elem_type
            })), SpvStorageClassFunction);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case lea_op: {
            SpvId base = emit_value(emitter, args.nodes[0]);

            LARRAY(SpvId, indices, args.count - 2);
            for (size_t i = 2; i < args.count; i++)
                indices[i - 2] = args.nodes[i] ? emit_value(emitter, args.nodes[i]) : 0;

            if (args.nodes[1]) {
                error("TODO: OpPtrAccessChain")
            } else {
                const Type* target_type = instr->type;
                SpvId result = spvb_access_chain(bb_builder, emit_type(emitter, target_type), base, args.count - 2, indices);
                register_result(emitter, variables.nodes[0], result);
            }
            return;
        }
        case select_op: {
            SpvId cond = emit_value(emitter, args.nodes[0]);
            SpvId truv = emit_value(emitter, args.nodes[1]);
            SpvId flsv = emit_value(emitter, args.nodes[2]);

            SpvId result = spvb_select(bb_builder, emit_type(emitter, variables.nodes[0]->type), cond, truv, flsv);
            register_result(emitter, variables.nodes[0], result);
            return;
        }
        case debug_printf_op: {
            assert(args.count >= 1);
            LARRAY(SpvId, arr, args.count);
            arr[0] = spvb_debug_string(emitter->file_builder, extract_string_literal(args.nodes[0]));
            for (size_t i = 1; i < args.count; i++)
                arr[i] = emit_value(emitter, args.nodes[i]);
            spvb_ext_instruction(bb_builder, emit_type(emitter, unit_type(emitter->arena)), emitter->non_semantic_imported_instrs.debug_printf, NonSemanticDebugPrintfDebugPrintf, args.count, arr);
            return;
        }
        default: error("TODO: unhandled op");
    }
    SHADY_UNREACHABLE;
}

static void emit_call(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, BBBuilder bb_builder, Call call, size_t results_count, SpvId results[]) {
    const Type* callee_type = call.callee->type;
    if (call.is_indirect) {
        callee_type = extract_operand_type(call.callee->type);
        assert(callee_type->tag == PtrType_TAG);
        callee_type = callee_type->payload.ptr_type.pointed_type;
    }
    assert(callee_type->tag == FnType_TAG);
    SpvId return_type = nodes_to_codom(emitter, callee_type->payload.fn_type.return_types);
    SpvId callee = emit_value(emitter, call.callee);
    LARRAY(SpvId, args, call.args.count);
    for (size_t i = 0; i < call.args.count; i++)
        args[i] = emit_value(emitter, call.args.nodes[i]);
    SpvId result = spvb_call(bb_builder, return_type, callee, call.args.count, args);
    switch (variables.count) {
        case 0: break;
        case 1: {
            register_result(emitter, variables.nodes[0], result);
            break;
        }
        default: {
            for (size_t i = 0; i < variables.count; i++) {
                SpvId result_type = emit_type(emitter, variables.nodes[i]->type);
                SpvId extracted_component = spvb_extract(bb_builder, result_type, result, 1, (uint32_t []) { i });
                register_result(emitter, variables.nodes[i], extracted_component);
            }
            break;
        }
    }
}

static void emit_if(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, If if_instr, size_t results_count, SpvId results[]) {
    SpvId join_bb_id = spvb_fresh_id(emitter->file_builder);

    SpvId true_id = spvb_fresh_id(emitter->file_builder);
    SpvId false_id = if_instr.if_false ? spvb_fresh_id(emitter->file_builder) : join_bb_id;

    spvb_selection_merge(*bb_builder, join_bb_id, 0);
    SpvId condition = emit_value(emitter, if_instr.condition);
    spvb_branch_conditional(*bb_builder, condition, true_id, false_id);

    // When 'join' is codegen'd, these will be filled with the values given to it
    BBBuilder join_bb = spvb_begin_bb(fn_builder, join_bb_id);
    LARRAY(struct Phi*, join_phis, variables.count);
    for (size_t i = 0; i < variables.count; i++) {
        assert(if_instr.if_false && "Ifs with yield types need false branches !");
        SpvId phi_id = spvb_fresh_id(emitter->file_builder);
        SpvId type = emit_type(emitter, extract_operand_type(variables.nodes[i]->type));
        struct Phi* phi = spvb_add_phi(join_bb, type, phi_id);
        join_phis[i] = phi;
        register_result(emitter, variables.nodes[i], phi_id);
    }

    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.join_target = join_bb_id;
    merge_targets_branches.join_phis = join_phis;

    BBBuilder true_bb = spvb_begin_bb(fn_builder, true_id);
    spvb_add_bb(fn_builder, true_bb);
    emit_body(emitter, fn_builder, true_bb, merge_targets_branches, if_instr.if_true);
    if (if_instr.if_false) {
        BBBuilder false_bb = spvb_begin_bb(fn_builder, false_id);
        spvb_add_bb(fn_builder, false_bb);
        emit_body(emitter, fn_builder, false_bb, merge_targets_branches, if_instr.if_false);
    }

    spvb_add_bb(fn_builder, join_bb);
    *bb_builder = join_bb;
}

static void emit_match(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Match match, size_t results_count, SpvId results[]) {
    assert(match.yield_types.count == 0 && "TODO use phis");

    SpvId next_id = spvb_fresh_id(emitter->file_builder);

    assert(extract_operand_type(match.inspect->type)->tag == Int_TAG);
    SpvId inspectee = emit_value(emitter, match.inspect);

    SpvId default_id = spvb_fresh_id(emitter->file_builder);
    LARRAY(SpvId, literals_and_cases, match.cases.count * 2);
    for (size_t i = 0; i < match.cases.count; i++) {
        literals_and_cases[i * 2 + 0] = (SpvId) (uint32_t) extract_int_literal_value(match.literals.nodes[i], true);
        literals_and_cases[i * 2 + 1] = spvb_fresh_id(emitter->file_builder);
    }

    spvb_selection_merge(*bb_builder, next_id, 0);
    spvb_switch(*bb_builder, inspectee, default_id, match.cases.count, literals_and_cases);

    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.join_target = next_id;

    for (size_t i = 0; i < match.cases.count; i++) {
        BBBuilder case_bb = spvb_begin_bb(fn_builder, literals_and_cases[i * 2 + 1]);
        emit_body(emitter, fn_builder, case_bb, merge_targets_branches, match.cases.nodes[i]);
        spvb_add_bb(fn_builder, case_bb);
    }
    BBBuilder default_bb = spvb_begin_bb(fn_builder, default_id);
    emit_body(emitter, fn_builder, default_bb, merge_targets_branches, match.default_case);
    spvb_add_bb(fn_builder, default_bb);

    assert(variables.count == 0 && "TODO implement variables using phi nodes");

    BBBuilder next = spvb_begin_bb(fn_builder, next_id);
    spvb_add_bb(fn_builder, next);
    *bb_builder = next;
}

static void emit_loop(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Loop loop_instr, size_t results_count, SpvId results[]) {
    assert(loop_instr.yield_types.count == variables.count);

    // First we create all the basic blocks we'll need
    SpvId header_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder header_builder = spvb_begin_bb(fn_builder, header_id);
    spvb_name(emitter->file_builder, header_id, "loop_header");

    SpvId body_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder body_builder = spvb_begin_bb(fn_builder, body_id);
    spvb_name(emitter->file_builder, body_id, "loop_body");

    SpvId continue_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder continue_builder = spvb_begin_bb(fn_builder, continue_id);
    spvb_name(emitter->file_builder, continue_id, "loop_continue");

    SpvId next_id = spvb_fresh_id(emitter->file_builder);
    BBBuilder next = spvb_begin_bb(fn_builder, next_id);
    spvb_name(emitter->file_builder, next_id, "loop_next");

    // Wire up the phi nodes for loop exit
    LARRAY(struct Phi*, loop_break_phis, loop_instr.params.count);
    for (size_t i = 0; i < loop_instr.yield_types.count; i++) {
        SpvId yielded_type = emit_type(emitter, extract_operand_type(loop_instr.yield_types.nodes[i]));

        SpvId break_phi_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* phi = spvb_add_phi(next, yielded_type, break_phi_id);
        loop_break_phis[i] = phi;
        register_result(emitter, variables.nodes[i], break_phi_id);
    }

    // Wire up the phi nodes for the loop contents
    LARRAY(struct Phi*, loop_continue_phis, loop_instr.params.count);
    for (size_t i = 0; i < loop_instr.params.count; i++) {
        SpvId loop_param_type = emit_type(emitter, extract_operand_type(loop_instr.params.nodes[i]->type));

        SpvId continue_phi_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* continue_phi = spvb_add_phi(continue_builder, loop_param_type, continue_phi_id);
        loop_continue_phis[i] = continue_phi;

        // To get the actual loop parameter, we make a second phi for the nodes that go into the header
        // We already know the two edges into the header so we immediately add the Phi sources for it.
        SpvId loop_param_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* loop_param_phi = spvb_add_phi(header_builder, loop_param_type, loop_param_id);
        SpvId param_initial_value = emit_value(emitter, loop_instr.initial_args.nodes[i]);
        spvb_add_phi_source(loop_param_phi, get_block_builder_id(*bb_builder), param_initial_value);
        spvb_add_phi_source(loop_param_phi, get_block_builder_id(continue_builder), continue_phi_id);
        register_result(emitter, loop_instr.params.nodes[i], loop_param_id);
    }

    // The current block goes to the header (it can't be the header itself !)
    spvb_branch(*bb_builder, header_id);
    spvb_add_bb(fn_builder, header_builder);

    // the header block receives the loop merge annotation
    spvb_loop_merge(header_builder, next_id, continue_id, 0, 0, NULL);
    spvb_branch(header_builder, body_id);
    spvb_add_bb(fn_builder, body_builder);

    // Emission of the body requires extra info for the break/continue merge terminators
    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.continue_target = continue_id;
    merge_targets_branches.continue_phis = loop_continue_phis;
    merge_targets_branches.break_target = next_id;
    merge_targets_branches.break_phis = loop_break_phis;
    emit_body(emitter, fn_builder, body_builder, merge_targets_branches, loop_instr.body);

    // the continue block just jumps back into the header
    spvb_branch(continue_builder, header_id);
    spvb_add_bb(fn_builder, continue_builder);

    // We start the next block
    spvb_add_bb(fn_builder, next);
    *bb_builder = next;
}

static void emit_instruction(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, const Node* instruction, size_t results_count, SpvId results[]) {
    assert(is_instruction(instruction));

    switch (instruction->tag) {
        case PrimOp_TAG: emit_primop(emitter, fn_builder, *bb_builder, instruction, results_count, results);                                    break;
        case Call_TAG:     emit_call(emitter, fn_builder, *bb_builder, instruction->payload.call_instr, results_count, results);                break;
        case If_TAG:         emit_if(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.if_instr, results_count, results);    break;
        case Match_TAG:   emit_match(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.match_instr, results_count, results); break;
        case Loop_TAG:     emit_loop(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.loop_instr, results_count, results);  break;
        default: error("Unrecognised instruction %s", node_tags[instruction->tag]);
    }
}

static SpvId find_reserved_id(Emitter* emitter, const Node* node) {
    SpvId* found = find_value_dict(const Node*, SpvId, emitter->node_ids, node);
    assert(found);
    return *found;
}

static BBBuilder find_basic_block_builder(Emitter* emitter, FnBuilder fn_builder, const Node* bb) {
    assert(is_basic_block(bb));
    BBBuilder* found = find_value_dict(const Node*, BBBuilder, emitter->bb_builders, bb);
    assert(found);
    return *found;
}

static void add_branch_phis(Emitter* emitter, FnBuilder fn_builder, SpvId src_id, const Node* dst, size_t count, SpvId args[]) {
    BBBuilder dst_builder = find_basic_block_builder(emitter, fn_builder, dst);
    struct List* phis = spbv_get_phis(dst_builder);
    assert(entries_count_list(phis) == count);
    for (size_t i = 0; i < count; i++) {
        struct Phi* phi = read_list(struct Phi*, phis)[i];
        spvb_add_phi_source(phi, src_id, args[i]);
    }
}

void emit_terminator(Emitter* emitter, FnBuilder fn_builder, BBBuilder basic_block_builder, MergeTargets merge_targets, const Node* terminator) {
    switch (is_terminator(terminator->tag)) {
        case Return_TAG: {
            const Nodes* ret_values = &terminator->payload.fn_ret.values;
            switch (ret_values->count) {
                case 0: spvb_return_void(basic_block_builder); return;
                case 1: spvb_return_value(basic_block_builder, emit_value(emitter, ret_values->nodes[0])); return;
                default: {
                    LARRAY(SpvId, arr, ret_values->count);
                    for (size_t i = 0; i < ret_values->count; i++)
                        arr[i] = emit_value(emitter, ret_values->nodes[i]);
                    SpvId return_that = spvb_composite(basic_block_builder, fn_ret_type_id(fn_builder), ret_values->count, arr);
                    spvb_return_value(basic_block_builder, return_that);
                    return;
                }
            }
        }
        case Let_TAG: {
            const Node* tail = terminator->payload.let.tail;
            assert(tail->tag == Lambda_TAG);
            Nodes params = tail->payload.lam.params;
            LARRAY(SpvId, results, params.count);
            emit_instruction(emitter, fn_builder, &basic_block_builder, &merge_targets, terminator->payload.let.instruction, params.count, results);
            if (tail->payload.lam.tier == Lambda_TAG) {
                for (size_t i = 0; i < params.count; i++)
                    register_result(emitter, params.nodes[i], results[i]);
                emit_terminator(emitter, fn_builder, basic_block_builder, merge_targets, tail->payload.lam.body);
            } else {
                spvb_branch(basic_block_builder, find_reserved_id(emitter, tail));
                add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), tail, params.count, results);
            }
            return;
        }
        case Branch_TAG: {
            Nodes args = terminator->payload.branch.args;
            LARRAY(SpvId, emitted_args, args.count);
            for (size_t i = 0; i < args.count; i++)
                emitted_args[i] = emit_value(emitter, args.nodes[i]);

            switch (terminator->payload.branch.branch_mode) {
                case BrJump: {
                    spvb_branch(basic_block_builder, find_reserved_id(emitter, terminator->payload.branch.target));
                    add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.branch.target, args.count, emitted_args);
                    return;
                }
                case BrIfElse: {
                    SpvId condition = emit_value(emitter, terminator->payload.branch.branch_condition);
                    spvb_branch_conditional(basic_block_builder, condition, find_reserved_id(emitter, terminator->payload.branch.true_target), find_reserved_id(emitter, terminator->payload.branch.false_target));
                    
                    add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.branch.true_target, args.count, emitted_args);
                    add_branch_phis(emitter, fn_builder, get_block_builder_id(basic_block_builder), terminator->payload.branch.false_target, args.count, emitted_args);
                    return;
                }
                case BrSwitch: error("TODO");
                default: SHADY_UNREACHABLE;
            }
        }
        case TailCall_TAG:
        case Join_TAG: error("Lower me");
        case MergeConstruct_TAG: {
            Nodes args = terminator->payload.merge_construct.args;
            switch (terminator->payload.merge_construct.construct) {
                case Selection: {
                    for (size_t i = 0; i < args.count; i++)
                        spvb_add_phi_source(merge_targets.join_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, args.nodes[i]));
                    spvb_branch(basic_block_builder, merge_targets.join_target);
                    return;
                }
                case Continue: {
                    for (size_t i = 0; i < args.count; i++)
                        spvb_add_phi_source(merge_targets.continue_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, args.nodes[i]));
                    spvb_branch(basic_block_builder, merge_targets.continue_target);
                    return;
                }
                case Break: {
                    for (size_t i = 0; i < args.count; i++)
                        spvb_add_phi_source(merge_targets.break_phis[i], get_block_builder_id(basic_block_builder), emit_value(emitter, args.nodes[i]));
                    spvb_branch(basic_block_builder, merge_targets.break_target);
                    return;
                }
                default: error("Not a merge.")
            }
        }
        case Unreachable_TAG: {
            spvb_unreachable(basic_block_builder);
            return;
        }
        case NotATerminator: error("TODO: emit terminator %s", node_tags[terminator->tag]);
    }
    SHADY_UNREACHABLE;
}

static void emit_basic_block(Emitter* emitter, FnBuilder fn_builder, const CFNode* node, bool is_entry) {
    assert(node->location.head->tag == Lambda_TAG);
    const Node* bb_node = node->location.head;
    // Find the preassigned ID to this
    BBBuilder bb_builder = find_basic_block_builder(emitter, fn_builder, bb_node);
    SpvId bb_id = get_block_builder_id(bb_builder);
    spvb_add_bb(fn_builder, bb_builder);
    spvb_name(emitter->file_builder, bb_id, bb_node->payload.lam.name);

    MergeTargets merge_targets = {
        .continue_target = 0,
        .break_target = 0,
        .join_target = 0
    };

    emit_terminator(emitter, fn_builder, bb_builder, merge_targets, bb_node->payload.lam.body);

    // Emit the child nodes
    size_t dom_count = entries_count_list(node->dominates);
    for (size_t i = 0; i < dom_count; i++) {
        CFNode* child_node = read_list(CFNode*, node->dominates)[i];
        emit_basic_block(emitter, fn_builder, child_node, false);
    }
}

static void emit_function(Emitter* emitter, const Node* node) {
    assert(node->tag == Lambda_TAG);

    const Type* fn_type = node->type;
    FnBuilder fn_builder = spvb_begin_fn(emitter->file_builder, find_reserved_id(emitter, node), emit_type(emitter, fn_type), nodes_to_codom(emitter, node->payload.lam.return_types));

    Nodes params = node->payload.lam.params;
    for (size_t i = 0; i < params.count; i++) {
        SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, params.nodes[i]->payload.var.type));
        insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, params.nodes[i], param_id);
    }

    Scope scope = build_scope_from_basic_block(node);
    // reserve a bunch of identifiers for the basic blocks in the scope
    for (size_t i = 0; i < scope.size; i++) {
        CFNode* cfnode = read_list(CFNode*, scope.contents)[i];
        assert(cfnode);
        const Node* bb = cfnode->location.head;
        assert(is_basic_block(bb) || bb == node);
        SpvId bb_id = i == spvb_fresh_id(emitter->file_builder);
        BBBuilder basic_block_builder = spvb_begin_bb(fn_builder, bb_id);
        insert_dict(const Node*, BBBuilder, emitter->bb_builders, bb, basic_block_builder);
        // add phis for every non-entry basic block
        if (i > 0) {
            assert(is_basic_block(bb) && bb != node);
            Nodes bb_params = bb->payload.lam.params;
            for (size_t j = 0; j < bb_params.count; j++) {
                const Node* bb_param = bb_params.nodes[j];
                spvb_add_phi(basic_block_builder, emit_type(emitter, bb_param->type), spvb_fresh_id(emitter->file_builder));
            }
            // also make sure to register the label for basic blocks
            register_result(emitter, bb, bb_id);
        }
    }
    // emit the blocks using the dominator tree
    emit_basic_block(emitter, fn_builder, scope.entry, true);
    dispose_scope(&scope);

    spvb_define_function(emitter->file_builder, fn_builder);
}

static void emit_decl(Emitter* emitter, const Node* decl, SpvId given_id) {
    switch (decl->tag) {
        case GlobalVariable_TAG: {
            const GlobalVariable* gvar = &decl->payload.global_variable;
            SpvId init = 0;
            if (gvar->init)
                init = emit_value(emitter, gvar->init);
            SpvStorageClass storage_class = emit_addr_space(gvar->address_space);
            spvb_global_variable(emitter->file_builder, given_id, emit_type(emitter, decl->type), storage_class, false, init);
            spvb_name(emitter->file_builder, given_id, gvar->name);

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
                    size_t set     = extract_int_literal_value(extract_annotation_payload(descriptor_set),     false);
                    size_t binding = extract_int_literal_value(extract_annotation_payload(descriptor_binding), false);
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationDescriptorSet, 1, (uint32_t []) { set });
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationBinding, 1, (uint32_t []) { binding });
                    break;
                }
                default: break;
            }

            break;
        } case Lambda_TAG: {
            assert(is_function(decl));
            emit_function(emitter, decl);
            spvb_name(emitter->file_builder, given_id, decl->payload.lam.name);
            break;
        } case Constant_TAG: {
            // We don't emit constants at all !
            // With RefDecl, we directly grab the underlying value and emit that there and then.
            // Emitting constants as their own IDs would be nicer, but it's painful to do because decls need their ID to be reserved in advance,
            // but we also desire to cache reused values instead of emitting them multiple times. This means we can't really "force" an ID for a given value.
            // The ideal fix would be if SPIR-V offered a way to "alias" an ID under a new one. This would allow applying new debug information to the decl ID, separate from the other instances of that value.
            break;
        }
        default: error("unhandled declaration kind")
    }
}

static void emit_root(Emitter* emitter, const Node* root) {
    assert(root->tag == Root_TAG);
    Nodes declarations = root->payload.root.declarations;

     // First reserve IDs for declarations
    LARRAY(SpvId, ids, declarations.count);
    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        ids[i] = spvb_fresh_id(emitter->file_builder);
        insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, decl, ids[i]);
    }

    for (size_t i = 0; i < declarations.count; i++) {
        const Node* decl = declarations.nodes[i];
        emit_decl(emitter, decl, ids[i]);
    }
}

static SpvExecutionModel emit_exec_model(ExecutionModel model) {
    switch (model) {
        case Compute:  return SpvExecutionModelGLCompute;
        case Vertex:   return SpvExecutionModelVertex;
        case Fragment: return SpvExecutionModelFragment;
        default: error("Can't emit execution model");
    }
}

static void emit_entry_points(Emitter* emitter, const Node* root) {
    assert(root->tag == Root_TAG);
    Nodes declarations = root->payload.root.declarations;

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
        if (decl->tag != Lambda_TAG) continue;
        SpvId fn_id = find_reserved_id(emitter, decl);

        const Node* entry_point = lookup_annotation(decl, "EntryPoint");
        if (entry_point) {
            assert(entry_point->payload.annotation.payload_type == AnPayloadValue);
            const char* execution_model_name = extract_string_literal(entry_point->payload.annotation.value);
            SpvExecutionModel execution_model = emit_exec_model(execution_model_from_string(execution_model_name));

            spvb_entry_point(emitter->file_builder, execution_model, fn_id, decl->payload.lam.name, interface_size, interface_arr);
            emitter->num_entry_pts++;

            const Node* workgroup_size = lookup_annotation(decl, "WorkgroupSize");
            if (execution_model == SpvExecutionModelGLCompute)
                assert(workgroup_size);
            if (workgroup_size) {
                assert(workgroup_size->payload.annotation.payload_type == AnPayloadValues);
                Nodes values = workgroup_size->payload.annotation.values;
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

void emit_spirv(CompilerConfig* config, IrArena* arena, const Node* root_node, size_t* output_size, char** output) {
    struct List* words = new_list(uint32_t);

    FileBuilder file_builder = spvb_begin();
    spvb_set_version(file_builder, config->target_spirv_version.major, config->target_spirv_version.minor);

    Emitter emitter = {
        .configuration = config,
        .arena = arena,
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

    emit_root(&emitter, root_node);
    emit_entry_points(&emitter, root_node);

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
