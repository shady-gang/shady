#include "emit_spv.h"

#include "log.h"
#include "portability.h"

#include "../../type.h"

#include <assert.h>

#include "spirv/unified1/NonSemanticDebugPrintf.h"

enum OperandKind {
    Signed, Unsigned, Float, Logical, Ptr, Other, OperandKindsCount
};

enum ResultKind {
    Same, Bool, TyOperand
};

static enum OperandKind classify_operand_type(const Type* type) {
    assert(is_type(type) && !contains_qualified_type(type));

    switch (type->tag) {
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
    Nodes tys = prim_op.type_arguments;

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
        enum OperandKind op_class = classify_operand_type(extract_operand_type(args.nodes[0]->type));
        if (entry.i_sel_mechanism == FirstOp) {
            opcode = entry.fo[op_class];
        } else if (entry.i_sel_mechanism == FirstAndResult) {
            enum OperandKind return_t_class = classify_operand_type(tys.nodes[0]);
            opcode = entry.foar[op_class][return_t_class];
        } else SHADY_UNREACHABLE;

        if (opcode == SpvOpNop) {
            assert(results_count == 1);
            results[0] = arr[0];
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

        assert(results_count == 1);
        if (args.count == 1)
            results[0] = spvb_unop(bb_builder, opcode, emit_type(emitter, result_t), arr[0]);
        else if (args.count == 2)
            results[1] = spvb_binop(bb_builder, opcode, emit_type(emitter, result_t), arr[0], arr[1]);
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
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case subgroup_broadcast_first_op: {
            SpvId scope_subgroup = emit_value(emitter, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_broadcast_first(bb_builder, emit_type(emitter, extract_operand_type(args.nodes[0]->type)), emit_value(emitter, args.nodes[0]), scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case subgroup_local_id_op: {
            SpvId result_t = emit_type(emitter, get_vulkan_builtins_type(emitter->arena, VulkanBuiltinSubgroupLocalInvocationId));
            SpvId ptr = emit_builtin(emitter, VulkanBuiltinSubgroupLocalInvocationId);
            SpvId result = spvb_load(bb_builder, result_t, ptr, 0, NULL);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case subgroup_elect_first_op: {
            SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
            SpvId scope_subgroup = emit_value(emitter, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_elect(bb_builder, result_t, scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
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
            assert(results_count == 1);
            results[0] = result;
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
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case store_op: {
            assert(extract_operand_type(args.nodes[0]->type)->tag == PtrType_TAG);
            SpvId eptr = emit_value(emitter, args.nodes[0]);
            SpvId eval = emit_value(emitter, args.nodes[1]);
            spvb_store(bb_builder, eval, eptr, 0, NULL);
            assert(results_count == 0);
            return;
        }
        case alloca_logical_op: {
            const Type* elem_type = args.nodes[0];
            SpvId result = spvb_local_variable(fn_builder, emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunctionLogical,
                .pointed_type = elem_type
            })), SpvStorageClassFunction);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case lea_op: {
            SpvId base = emit_value(emitter, args.nodes[0]);

            LARRAY(SpvId, indices, args.count - 2);
            for (size_t i = 2; i < args.count; i++)
                indices[i - 2] = args.nodes[i] ? emit_value(emitter, args.nodes[i]) : 0;

            const IntLiteral* known_offset = resolve_to_literal(args.nodes[1]);
            if (known_offset && known_offset->value.i64 == 0) {
                const Type* target_type = instr->type;
                SpvId result = spvb_access_chain(bb_builder, emit_type(emitter, target_type), base, args.count - 2, indices);
                assert(results_count == 1);
                results[0] = result;
            } else {
                error("TODO: OpPtrAccessChain")
            }
            return;
        }
        case select_op: {
            SpvId cond = emit_value(emitter, args.nodes[0]);
            SpvId truv = emit_value(emitter, args.nodes[1]);
            SpvId flsv = emit_value(emitter, args.nodes[2]);

            SpvId result = spvb_select(bb_builder, emit_type(emitter, args.nodes[1]->type), cond, truv, flsv);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case debug_printf_op: {
            assert(args.count >= 1);
            LARRAY(SpvId, arr, args.count);
            arr[0] = spvb_debug_string(emitter->file_builder, extract_string_literal(args.nodes[0]));
            for (size_t i = 1; i < args.count; i++)
                arr[i] = emit_value(emitter, args.nodes[i]);
            spvb_ext_instruction(bb_builder, emit_type(emitter, unit_type(emitter->arena)), emitter->non_semantic_imported_instrs.debug_printf, NonSemanticDebugPrintfDebugPrintf, args.count, arr);
            assert(results_count == 0);
            return;
        }
        default: error("TODO: unhandled op");
    }
    SHADY_UNREACHABLE;
}

static void emit_call(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, BBBuilder bb_builder, Call call, size_t results_count, SpvId results[]) {
    const Type* callee_type = call.callee->type;
    callee_type = extract_operand_type(call.callee->type);
    assert(callee_type->tag == PtrType_TAG);
    callee_type = callee_type->payload.ptr_type.pointed_type;
    assert(callee_type->tag == FnType_TAG);
    Nodes return_types = callee_type->payload.fn_type.return_types;
    SpvId return_type = nodes_to_codom(emitter, return_types);
    SpvId callee = emit_value(emitter, call.callee);
    LARRAY(SpvId, args, call.args.count);
    for (size_t i = 0; i < call.args.count; i++)
        args[i] = emit_value(emitter, call.args.nodes[i]);
    SpvId result = spvb_call(bb_builder, return_type, callee, call.args.count, args);
    switch (results_count) {
        case 0: break;
        case 1: {
            results[0] = result;
            break;
        }
        default: {
            assert(return_types.count == results_count);
            for (size_t i = 0; i < results_count; i++) {
                SpvId result_type = emit_type(emitter, return_types.nodes[i]->type);
                SpvId extracted_component = spvb_extract(bb_builder, result_type, result, 1, (uint32_t []) { i });
                results[i] = extracted_component;
            }
            break;
        }
    }
}

static void emit_if(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, If if_instr, size_t results_count, SpvId results[]) {
    Nodes yield_types = if_instr.yield_types;
    assert(yield_types.count == results_count);
    SpvId join_bb_id = spvb_fresh_id(emitter->file_builder);

    SpvId true_id = spvb_fresh_id(emitter->file_builder);
    SpvId false_id = if_instr.if_false ? spvb_fresh_id(emitter->file_builder) : join_bb_id;

    spvb_selection_merge(*bb_builder, join_bb_id, 0);
    SpvId condition = emit_value(emitter, if_instr.condition);
    spvb_branch_conditional(*bb_builder, condition, true_id, false_id);

    // When 'join' is codegen'd, these will be filled with the values given to it
    BBBuilder join_bb = spvb_begin_bb(fn_builder, join_bb_id);
    LARRAY(struct Phi*, join_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        assert(if_instr.if_false && "Ifs with yield types need false branches !");
        SpvId phi_id = spvb_fresh_id(emitter->file_builder);
        SpvId type = emit_type(emitter, extract_operand_type(yield_types.nodes[i]));
        struct Phi* phi = spvb_add_phi(join_bb, type, phi_id);
        join_phis[i] = phi;
        results[i] = phi_id;
    }

    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.join_target = join_bb_id;
    merge_targets_branches.join_phis = join_phis;

    BBBuilder true_bb = spvb_begin_bb(fn_builder, true_id);
    spvb_add_bb(fn_builder, true_bb);
    assert(is_anonymous_lambda(if_instr.if_true));
    emit_terminator(emitter, fn_builder, true_bb, merge_targets_branches, if_instr.if_true->payload.anon_lam.body);
    if (if_instr.if_false) {
        BBBuilder false_bb = spvb_begin_bb(fn_builder, false_id);
        spvb_add_bb(fn_builder, false_bb);
        assert(is_anonymous_lambda(if_instr.if_false));
        emit_terminator(emitter, fn_builder, false_bb, merge_targets_branches, if_instr.if_false->payload.anon_lam.body);
    }

    spvb_add_bb(fn_builder, join_bb);
    *bb_builder = join_bb;
}

static void emit_match(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Match match, size_t results_count, SHADY_UNUSED SpvId results[]) {
    assert(match.yield_types.count == 0 && "TODO use phis");
    assert(results_count == match.yield_types.count);

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
        const Node* case_body = match.cases.nodes[i];
        assert(is_anonymous_lambda(case_body));
        emit_terminator(emitter, fn_builder, case_bb, merge_targets_branches, case_body->payload.anon_lam.body);
        spvb_add_bb(fn_builder, case_bb);
    }
    BBBuilder default_bb = spvb_begin_bb(fn_builder, default_id);
    assert(is_anonymous_lambda(match.default_case));
    emit_terminator(emitter, fn_builder, default_bb, merge_targets_branches, match.default_case->payload.anon_lam.body);
    spvb_add_bb(fn_builder, default_bb);

    BBBuilder next = spvb_begin_bb(fn_builder, next_id);
    spvb_add_bb(fn_builder, next);
    *bb_builder = next;
}

static void emit_loop(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Loop loop_instr, size_t results_count, SpvId results[]) {
    Nodes yield_types = loop_instr.yield_types;
    assert(yield_types.count == results_count);

    const Node* body = loop_instr.body;
    assert(is_anonymous_lambda(body));
    Nodes body_params = body->payload.anon_lam.params;

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
    LARRAY(struct Phi*, loop_break_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        SpvId yielded_type = emit_type(emitter, extract_operand_type(yield_types.nodes[i]));

        SpvId break_phi_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* phi = spvb_add_phi(next, yielded_type, break_phi_id);
        loop_break_phis[i] = phi;
        results[i] = break_phi_id;
    }

    // Wire up the phi nodes for the loop contents
    LARRAY(struct Phi*, loop_continue_phis, body_params.count);
    for (size_t i = 0; i < body_params.count; i++) {
        SpvId loop_param_type = emit_type(emitter, extract_operand_type(body_params.nodes[i]->type));

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
        register_result(emitter, body_params.nodes[i], loop_param_id);
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
    emit_terminator(emitter, fn_builder, body_builder, merge_targets_branches, body->payload.anon_lam.body);

    // the continue block just jumps back into the header
    spvb_branch(continue_builder, header_id);
    spvb_add_bb(fn_builder, continue_builder);

    // We start the next block
    spvb_add_bb(fn_builder, next);
    *bb_builder = next;
}

void emit_instruction(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, const Node* instruction, size_t results_count, SpvId results[]) {
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