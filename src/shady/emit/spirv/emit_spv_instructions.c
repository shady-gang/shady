#include "emit_spv.h"

#include "log.h"
#include "portability.h"

#include "../../type.h"

#include <assert.h>

#include "spirv/unified1/NonSemanticDebugPrintf.h"

typedef enum {
    Signed, Unsigned, Float, Logical, Ptr, Other, OperandClassCount
} OperandClass;

typedef enum {
    Custom, BinOp, UnOp, Builtin
} InstrClass;

typedef enum {
    Same, Bool, TyOperand
} ResultClass;

static OperandClass classify_operand_type(const Type* type) {
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
typedef enum {
    None, One, FirstOp, FirstAndResult
} ISelMechanism;

#define ISEL_IDENTITY (SpvOpNop /* no-op, should be lowered to nothing beforehand */)
#define ISEL_LOWERME (SpvOpMax /* boolean conversions don't exist as a single instruction, a pass should lower them instead */)
#define ISEL_ILLEGAL (SpvOpMax /* doesn't make sense to support */)
#define ISEL_CUSTOM (SpvOpMax /* doesn't make sense to support */)

const struct IselTableEntry {
    InstrClass class;
    ISelMechanism isel_mechanism;
    ResultClass result_kind;
    union {
        SpvOp one;
        // matches first operand
        SpvOp fo[OperandClassCount];
        // matches first operand and return type [first operand][result type]
        SpvOp foar[OperandClassCount][OperandClassCount];
        VulkanBuiltins builtin;
    };
} isel_table[] = {
    [add_op] = { BinOp, FirstOp, Same, .fo = { SpvOpIAdd, SpvOpIAdd, SpvOpFAdd }},
    [sub_op] = { BinOp, FirstOp, Same, .fo = { SpvOpISub, SpvOpISub, SpvOpFSub }},
    [mul_op] = { BinOp, FirstOp, Same, .fo = { SpvOpIMul, SpvOpIMul, SpvOpFMul }},
    [div_op] = { BinOp, FirstOp, Same, .fo = { SpvOpSDiv, SpvOpUDiv, SpvOpFDiv }},
    [mod_op] = { BinOp, FirstOp, Same, .fo = { SpvOpSMod, SpvOpUMod, SpvOpFMod }},

    [neg_op] = { UnOp, FirstOp, Same, .fo = { SpvOpSNegate, SpvOpSNegate }},

    [eq_op]  = { BinOp, FirstOp, Bool, .fo = { SpvOpIEqual,            SpvOpIEqual,            SpvOpFOrdNotEqual,         SpvOpLogicalEqual    }},
    [neq_op] = { BinOp, FirstOp, Bool, .fo = { SpvOpINotEqual,         SpvOpINotEqual,         SpvOpFOrdNotEqual,         SpvOpLogicalNotEqual }},
    [lt_op]  = { BinOp, FirstOp, Bool, .fo = { SpvOpSLessThan,         SpvOpULessThan,         SpvOpFOrdLessThan,         ISEL_IDENTITY        }},
    [lte_op] = { BinOp, FirstOp, Bool, .fo = { SpvOpSLessThanEqual,    SpvOpULessThanEqual,    SpvOpFOrdLessThanEqual,    ISEL_IDENTITY        }},
    [gt_op]  = { BinOp, FirstOp, Bool, .fo = { SpvOpSGreaterThan,      SpvOpUGreaterThan,      SpvOpFOrdGreaterThan,      ISEL_IDENTITY        }},
    [gte_op] = { BinOp, FirstOp, Bool, .fo = { SpvOpSGreaterThanEqual, SpvOpUGreaterThanEqual, SpvOpFOrdGreaterThanEqual, ISEL_IDENTITY        }},

    [not_op] = { UnOp, FirstOp, Same, .fo = { SpvOpNot,        SpvOpNot,        SpvOpLogicalNot      }},

    [and_op] = { BinOp, FirstOp, Same, .fo = { SpvOpBitwiseAnd, SpvOpBitwiseAnd, SpvOpLogicalAnd      }},
    [or_op]  = { BinOp, FirstOp, Same, .fo = { SpvOpBitwiseOr,  SpvOpBitwiseOr,  SpvOpLogicalOr       }},
    [xor_op] = { BinOp, FirstOp, Same, .fo = { SpvOpBitwiseXor, SpvOpBitwiseXor, SpvOpLogicalNotEqual }},

    [lshift_op]         = { BinOp, FirstOp, Same, .fo = { SpvOpShiftLeftLogical,     SpvOpShiftLeftLogical,     ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_arithm_op]  = { BinOp, FirstOp, Same, .fo = { SpvOpShiftRightArithmetic, SpvOpShiftRightArithmetic, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_logical_op] = { BinOp, FirstOp, Same, .fo = { SpvOpShiftRightLogical,    SpvOpShiftRightLogical,    ISEL_ILLEGAL, ISEL_ILLEGAL }},

    [convert_op] = { UnOp, FirstAndResult, TyOperand, .foar = {
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertSToF, ISEL_LOWERME  },
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertUToF, ISEL_LOWERME  },
        { SpvOpConvertFToS, SpvOpConvertFToU, SpvOpFConvert,    ISEL_ILLEGAL  },
        { ISEL_LOWERME,     ISEL_LOWERME,     ISEL_ILLEGAL,     ISEL_IDENTITY }
    }},

    [reinterpret_op] = { UnOp, FirstAndResult, TyOperand, .foar = {
        { SpvOpUConvert,      SpvOpBitcast,       SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       ISEL_IDENTITY,      SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       SpvOpBitcast,       ISEL_IDENTITY, ISEL_ILLEGAL,  ISEL_ILLEGAL /* no fp-ptr casts */ },
        { ISEL_ILLEGAL,       ISEL_ILLEGAL,       ISEL_ILLEGAL,  ISEL_IDENTITY, ISEL_ILLEGAL /* no bool reinterpret */ },
        { SpvOpConvertPtrToU, SpvOpConvertPtrToU, ISEL_ILLEGAL,  ISEL_ILLEGAL,  ISEL_IDENTITY }
    }},

    [subgroup_local_id_op] = { Builtin, .builtin = VulkanBuiltinSubgroupLocalInvocationId },
    [subgroup_id_op] = { Builtin, .builtin = VulkanBuiltinSubgroupId },
    [workgroup_local_id_op] = { Builtin, .builtin = VulkanBuiltinLocalInvocationId },
    [workgroup_num_op] = { Builtin, .builtin = VulkanBuiltinNumWorkgroups },
    [workgroup_id_op] = { Builtin, .builtin = VulkanBuiltinWorkgroupId },
    [workgroup_size_op] = { Builtin, .builtin = VulkanBuiltinWorkgroupSize },
    [global_id_op] = { Builtin, .builtin = VulkanBuiltinGlobalInvocationId },

    [PRIMOPS_COUNT] = { Custom }
};

static const Type* get_result_t(Emitter* emitter, struct IselTableEntry entry, Nodes args, Nodes type_arguments) {
    switch (entry.result_kind) {
        case Same:      return get_unqualified_type(first(args)->type);
        case Bool:      return bool_type(emitter->arena);
        case TyOperand: return first(type_arguments);
        default: error("unhandled result kind");
    }
}

static SpvOp get_opcode(Emitter* emitter, struct IselTableEntry entry, Nodes args, Nodes type_arguments) {
    OperandClass op_class = classify_operand_type(get_unqualified_type(first(args)->type));
    switch (entry.isel_mechanism) {
        case None:    return SpvOpMax;
        case One:     return entry.one;
        case FirstOp: return entry.fo[op_class];
        case FirstAndResult: {
            assert(type_arguments.count == 1);
            OperandClass return_t_class = classify_operand_type(first(type_arguments));
            return entry.foar[op_class][return_t_class];
        }
    }
}

static void emit_primop(Emitter* emitter, FnBuilder fn_builder, BBBuilder bb_builder, const Node* instr, size_t results_count, SpvId results[]) {
    PrimOp prim_op = instr->payload.prim_op;
    Nodes args = prim_op.operands;
    Nodes type_arguments = prim_op.type_arguments;

    struct IselTableEntry entry = isel_table[prim_op.op];
    if (entry.class != Custom) {
        LARRAY(SpvId, emitted_args, args.count);
        for (size_t i = 0; i < args.count; i++)
            emitted_args[i] = emit_value(emitter, bb_builder, args.nodes[i]);

        switch (entry.class) {
            case UnOp: {
                assert(args.count == 1 && results_count == 1);
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                if (opcode == SpvOpNop) {
                    assert(results_count == 1);
                    results[0] = emitted_args[0];
                    return;
                }
                assert(opcode != SpvOpMax);
                const Type* result_t = get_result_t(emitter, entry, args, type_arguments);
                results[0] = spvb_unop(bb_builder, opcode, emit_type(emitter, result_t), emitted_args[0]);
                return;
            }
            case BinOp: {
                assert(args.count == 2 && results_count == 1);
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                assert(opcode != SpvOpMax);
                const Type* result_t = get_result_t(emitter, entry, args, type_arguments);
                results[0] = spvb_binop(bb_builder, opcode, emit_type(emitter, result_t), emitted_args[0], emitted_args[1]);
                return;
            }
            case Builtin: {
                assert(args.count == 0 && results_count == 1);
                SpvId result_t = emit_type(emitter, get_vulkan_builtins_type(emitter->arena, entry.builtin));
                SpvId ptr = emit_builtin(emitter, entry.builtin);
                SpvId result = spvb_load(bb_builder, result_t, ptr, 0, NULL);
                results[0] = result;
                return;
            }
            case Custom: SHADY_UNREACHABLE;
        }

        return;
    }
    switch (prim_op.op) {
        case subgroup_ballot_op: {
            const Type* i32x4 = pack_type(emitter->arena, (PackType) { .width = 4, .element_type = int32_type(emitter->arena) });
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_ballot(bb_builder, emit_type(emitter, i32x4), emit_value(emitter, bb_builder, first(args)), scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case subgroup_broadcast_first_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_broadcast_first(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case subgroup_reduce_sum_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            assert(results_count == 1);
            results[0] = spvb_non_uniform_iadd(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup, SpvGroupOperationReduce, NULL);
            return;
        }
        case subgroup_elect_first_op: {
            SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_elect(bb_builder, result_t, scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case extract_op: {
            const Node* src_value = first(args);
            const Type* result_t = instr->type;
            LARRAY(uint32_t, arr, args.count - 1);
            for (size_t i = 0; i < args.count - 1; i++) {
                arr[i] = get_int_literal_value(args.nodes[i + 1], false);
            }
            assert(args.count > 1);
            SpvId result = spvb_extract(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, src_value), args.count - 1, arr);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case make_op: {
            const Node* src = first(args);
            SpvId src_id = emit_value(emitter, bb_builder, src);
            const Type* dst_type = first(type_arguments);
            assert(dst_type->tag == TypeDeclRef_TAG);
            const Node* nom_decl = dst_type->payload.type_decl_ref.decl;
            assert(nom_decl->tag == NominalType_TAG);
            const Node* backing_type = nom_decl->payload.nom_type.body;
            assert(backing_type->tag == RecordType_TAG);
            switch (backing_type->tag) {
                case RecordType_TAG: {
                    Nodes components = backing_type->payload.record_type.members;
                    LARRAY(SpvId, extracted, components.count);
                    for (size_t i = 0; i < components.count; i++) {
                        extracted[i] = spvb_extract(bb_builder, emit_type(emitter, components.nodes[i]), src_id, 1, (uint32_t[]) { i });
                    }
                    results[0] = spvb_composite(bb_builder, emit_type(emitter, dst_type), components.count, extracted);
                    break;
                }
                case NotAType: assert(false);
                default: error("unhandled backing type");
            }
            return;
        }
        case load_op: {
            assert(get_unqualified_type(first(args)->type)->tag == PtrType_TAG);
            const Type* elem_type = get_unqualified_type(first(args)->type)->payload.ptr_type.pointed_type;
            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            SpvId result = spvb_load(bb_builder, emit_type(emitter, elem_type), eptr, 0, NULL);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case store_op: {
            assert(get_unqualified_type(first(args)->type)->tag == PtrType_TAG);
            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            SpvId eval = emit_value(emitter, bb_builder, args.nodes[1]);
            spvb_store(bb_builder, eval, eptr, 0, NULL);
            assert(results_count == 0);
            return;
        }
        case alloca_logical_op: {
            const Type* elem_type = first(type_arguments);
            SpvId result = spvb_local_variable(fn_builder, emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunctionLogical,
                .pointed_type = elem_type
            })), SpvStorageClassFunction);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case lea_op: {
            SpvId base = emit_value(emitter, bb_builder, first(args));

            LARRAY(SpvId, indices, args.count - 2);
            for (size_t i = 2; i < args.count; i++)
                indices[i - 2] = args.nodes[i] ? emit_value(emitter, bb_builder, args.nodes[i]) : 0;

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
            SpvId cond = emit_value(emitter, bb_builder, first(args));
            SpvId truv = emit_value(emitter, bb_builder, args.nodes[1]);
            SpvId flsv = emit_value(emitter, bb_builder, args.nodes[2]);

            SpvId result = spvb_select(bb_builder, emit_type(emitter, args.nodes[1]->type), cond, truv, flsv);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case debug_printf_op: {
            assert(args.count >= 1);
            LARRAY(SpvId, arr, args.count);
            arr[0] = spvb_debug_string(emitter->file_builder, get_string_literal(emitter->arena, first(args)));
            for (size_t i = 1; i < args.count; i++)
                arr[i] = emit_value(emitter, bb_builder, args.nodes[i]);
            spvb_ext_instruction(bb_builder, emit_type(emitter, unit_type(emitter->arena)), emitter->non_semantic_imported_instrs.debug_printf, NonSemanticDebugPrintfDebugPrintf, args.count, arr);
            assert(results_count == 0);
            return;
        }
        default: error("TODO: unhandled op");
    }
    error("unreachable");
}

static void emit_leaf_call(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, BBBuilder bb_builder, LeafCall call, size_t results_count, SpvId results[]) {
    const Node* fn = call.callee;
    SpvId callee = emit_decl(emitter, fn);

    const Type* callee_type = call.callee->type;
    assert(callee_type->tag == FnType_TAG);
    Nodes return_types = callee_type->payload.fn_type.return_types;
    SpvId return_type = nodes_to_codom(emitter, return_types);
    LARRAY(SpvId, args, call.args.count);
    for (size_t i = 0; i < call.args.count; i++)
        args[i] = emit_value(emitter, bb_builder, call.args.nodes[i]);
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
    SpvId condition = emit_value(emitter, *bb_builder, if_instr.condition);
    spvb_branch_conditional(*bb_builder, condition, true_id, false_id);

    // When 'join' is codegen'd, these will be filled with the values given to it
    BBBuilder join_bb = spvb_begin_bb(fn_builder, join_bb_id);
    LARRAY(struct Phi*, join_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        assert(if_instr.if_false && "Ifs with yield types need false branches !");
        SpvId phi_id = spvb_fresh_id(emitter->file_builder);
        SpvId type = emit_type(emitter, get_unqualified_type(yield_types.nodes[i]));
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

    assert(get_unqualified_type(match.inspect->type)->tag == Int_TAG);
    SpvId inspectee = emit_value(emitter, *bb_builder, match.inspect);

    SpvId default_id = spvb_fresh_id(emitter->file_builder);
    LARRAY(SpvId, literals_and_cases, match.cases.count * 2);
    for (size_t i = 0; i < match.cases.count; i++) {
        literals_and_cases[i * 2 + 0] = (SpvId) (uint32_t) get_int_literal_value(match.literals.nodes[i], true);
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
        spvb_add_bb(fn_builder, case_bb);
        emit_terminator(emitter, fn_builder, case_bb, merge_targets_branches, case_body->payload.anon_lam.body);
    }
    BBBuilder default_bb = spvb_begin_bb(fn_builder, default_id);
    assert(is_anonymous_lambda(match.default_case));
    spvb_add_bb(fn_builder, default_bb);
    emit_terminator(emitter, fn_builder, default_bb, merge_targets_branches, match.default_case->payload.anon_lam.body);

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
        SpvId yielded_type = emit_type(emitter, get_unqualified_type(yield_types.nodes[i]));

        SpvId break_phi_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* phi = spvb_add_phi(next, yielded_type, break_phi_id);
        loop_break_phis[i] = phi;
        results[i] = break_phi_id;
    }

    // Wire up the phi nodes for the loop contents
    LARRAY(struct Phi*, loop_continue_phis, body_params.count);
    for (size_t i = 0; i < body_params.count; i++) {
        SpvId loop_param_type = emit_type(emitter, get_unqualified_type(body_params.nodes[i]->type));

        SpvId continue_phi_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* continue_phi = spvb_add_phi(continue_builder, loop_param_type, continue_phi_id);
        loop_continue_phis[i] = continue_phi;

        // To get the actual loop parameter, we make a second phi for the nodes that go into the header
        // We already know the two edges into the header so we immediately add the Phi sources for it.
        SpvId loop_param_id = spvb_fresh_id(emitter->file_builder);
        struct Phi* loop_param_phi = spvb_add_phi(header_builder, loop_param_type, loop_param_id);
        SpvId param_initial_value = emit_value(emitter, *bb_builder, loop_instr.initial_args.nodes[i]);
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
        case IndirectCall_TAG: error("SPIR-V does not support indirect function calls. Such instructions need to be lowered away.");
        case LeafCall_TAG: emit_leaf_call(emitter, fn_builder, *bb_builder, instruction->payload.leaf_call, results_count, results);                 break;
        case PrimOp_TAG:      emit_primop(emitter, fn_builder, *bb_builder, instruction, results_count, results);                                    break;
        case If_TAG:              emit_if(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.if_instr, results_count, results);    break;
        case Match_TAG:        emit_match(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.match_instr, results_count, results); break;
        case Loop_TAG:          emit_loop(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.loop_instr, results_count, results);  break;
        default: error("Unrecognised instruction %s", node_tags[instruction->tag]);
    }
}