#include "emit_spv.h"

#include "log.h"
#include "dict.h"
#include "portability.h"

#include "../shady/type.h"
#include "../shady/transform/memory_layout.h"
#include "../shady/transform/ir_gen_helpers.h"
#include "../shady/analysis/cfg.h"
#include "../shady/analysis/scheduler.h"

#include <assert.h>
#include <string.h>

#include "spirv/unified1/NonSemanticDebugPrintf.h"
#include "spirv/unified1/GLSL.std.450.h"

typedef enum {
    Custom, Plain,
} InstrClass;

/// What is considered when searching for an instruction
typedef enum {
    None, Monomorphic, FirstOp, FirstAndResult
} ISelMechanism;

typedef enum {
    Same, SameTuple, Bool, Void, TyOperand
} ResultClass;

typedef enum {
    Signed, Unsigned, FP, Logical, Ptr, OperandClassCount
} OperandClass;

static OperandClass classify_operand_type(const Type* type) {
    assert(is_type(type) && is_data_type(type));

    if (type->tag == PackType_TAG)
        return classify_operand_type(type->payload.pack_type.element_type);

    switch (type->tag) {
        case Int_TAG:     return type->payload.int_type.is_signed ? Signed : Unsigned;
        case Bool_TAG:    return Logical;
        case PtrType_TAG: return Ptr;
        case Float_TAG:   return FP;
        default: error("we don't know what to do with this")
    }
}

typedef struct  {
    InstrClass class;
    ISelMechanism isel_mechanism;
    ResultClass result_kind;
    union {
        SpvOp op;
        // matches first operand
        SpvOp fo[OperandClassCount];
        // matches first operand and return type [first operand][result type]
        SpvOp foar[OperandClassCount][OperandClassCount];
    };
    const char* extended_set;
} IselTableEntry;

#define ISEL_IDENTITY (SpvOpNop /* no-op, should be lowered to nothing beforehand */)
#define ISEL_LOWERME (SpvOpMax /* boolean conversions don't exist as a single instruction, a pass should lower them instead */)
#define ISEL_ILLEGAL (SpvOpMax /* doesn't make sense to support */)
#define ISEL_CUSTOM (SpvOpMax /* doesn't make sense to support */)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

static const IselTableEntry isel_table[] = {
    [add_op] = {Plain, FirstOp, Same, .fo = {SpvOpIAdd, SpvOpIAdd, SpvOpFAdd, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [sub_op] = {Plain, FirstOp, Same, .fo = {SpvOpISub, SpvOpISub, SpvOpFSub, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [mul_op] = {Plain, FirstOp, Same, .fo = {SpvOpIMul, SpvOpIMul, SpvOpFMul, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [div_op] = {Plain, FirstOp, Same, .fo = {SpvOpSDiv, SpvOpUDiv, SpvOpFDiv, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [mod_op] = {Plain, FirstOp, Same, .fo = {SpvOpSMod, SpvOpUMod, SpvOpFMod, ISEL_ILLEGAL, ISEL_ILLEGAL }},

    [add_carry_op] = {Plain, FirstOp, SameTuple, .fo = {SpvOpIAddCarry, SpvOpIAddCarry, ISEL_ILLEGAL }},
    [sub_borrow_op] = {Plain, FirstOp, SameTuple, .fo = {SpvOpISubBorrow, SpvOpISubBorrow, ISEL_ILLEGAL }},
    [mul_extended_op] = {Plain, FirstOp, SameTuple, .fo = {SpvOpSMulExtended, SpvOpUMulExtended, ISEL_ILLEGAL }},

    [neg_op] = {Plain, FirstOp, Same, .fo = {SpvOpSNegate, SpvOpSNegate, SpvOpFNegate }},

    [eq_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpIEqual, SpvOpIEqual, SpvOpFOrdEqual, SpvOpLogicalEqual, SpvOpPtrEqual }},
    [neq_op] = {Plain, FirstOp, Bool, .fo = {SpvOpINotEqual, SpvOpINotEqual, SpvOpFOrdNotEqual, SpvOpLogicalNotEqual, SpvOpPtrNotEqual }},
    [lt_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpSLessThan, SpvOpULessThan, SpvOpFOrdLessThan, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [lte_op] = {Plain, FirstOp, Bool, .fo = {SpvOpSLessThanEqual, SpvOpULessThanEqual, SpvOpFOrdLessThanEqual, ISEL_ILLEGAL, ISEL_ILLEGAL}},
    [gt_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpSGreaterThan, SpvOpUGreaterThan, SpvOpFOrdGreaterThan, ISEL_ILLEGAL, ISEL_ILLEGAL}},
    [gte_op] = {Plain, FirstOp, Bool, .fo = {SpvOpSGreaterThanEqual, SpvOpUGreaterThanEqual, SpvOpFOrdGreaterThanEqual, ISEL_ILLEGAL, ISEL_ILLEGAL }},

    [not_op] = {Plain, FirstOp, Same, .fo = {SpvOpNot, SpvOpNot, ISEL_ILLEGAL, SpvOpLogicalNot }},

    [and_op] = {Plain, FirstOp, Same, .fo = {SpvOpBitwiseAnd, SpvOpBitwiseAnd, ISEL_ILLEGAL, SpvOpLogicalAnd      }},
    [or_op]  = {Plain, FirstOp, Same, .fo = {SpvOpBitwiseOr,  SpvOpBitwiseOr,  ISEL_ILLEGAL, SpvOpLogicalOr       }},
    [xor_op] = {Plain, FirstOp, Same, .fo = {SpvOpBitwiseXor, SpvOpBitwiseXor, ISEL_ILLEGAL, SpvOpLogicalNotEqual }},

    [lshift_op]         = {Plain, FirstOp, Same, .fo = {SpvOpShiftLeftLogical, SpvOpShiftLeftLogical, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_arithm_op]  = {Plain, FirstOp, Same, .fo = {SpvOpShiftRightArithmetic, SpvOpShiftRightArithmetic, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [rshift_logical_op] = {Plain, FirstOp, Same, .fo = {SpvOpShiftRightLogical, SpvOpShiftRightLogical, ISEL_ILLEGAL, ISEL_ILLEGAL }},

    [convert_op] = {Plain, FirstAndResult, TyOperand, .foar = {
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertSToF, ISEL_LOWERME,  ISEL_LOWERME  },
        { SpvOpSConvert,    SpvOpUConvert,    SpvOpConvertUToF, ISEL_LOWERME,  ISEL_LOWERME  },
        { SpvOpConvertFToS, SpvOpConvertFToU, SpvOpFConvert,    ISEL_ILLEGAL,  ISEL_ILLEGAL  },
        { ISEL_LOWERME,     ISEL_LOWERME,     ISEL_ILLEGAL,     ISEL_IDENTITY, ISEL_ILLEGAL  },
        { ISEL_LOWERME,     ISEL_LOWERME,     ISEL_ILLEGAL,     ISEL_ILLEGAL,  ISEL_IDENTITY }
    }},

    [reinterpret_op] = {Plain, FirstAndResult, TyOperand, .foar = {
        { ISEL_ILLEGAL,      SpvOpBitcast,       SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       ISEL_ILLEGAL,      SpvOpBitcast,  ISEL_ILLEGAL,  SpvOpConvertUToPtr },
        { SpvOpBitcast,       SpvOpBitcast,       ISEL_IDENTITY, ISEL_ILLEGAL,  ISEL_ILLEGAL /* no fp-ptr casts */ },
        { ISEL_ILLEGAL,       ISEL_ILLEGAL,       ISEL_ILLEGAL,  ISEL_IDENTITY, ISEL_ILLEGAL /* no bool reinterpret */ },
        { SpvOpConvertPtrToU, SpvOpConvertPtrToU, ISEL_ILLEGAL,  ISEL_ILLEGAL,  ISEL_CUSTOM }
    }},

    [sqrt_op] =     { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Sqrt },
    [inv_sqrt_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450InverseSqrt},
    [floor_op] =    { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Floor },
    [ceil_op] =     { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Ceil  },
    [round_op] =    { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Round },
    [fract_op] =    { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Fract },
    [sin_op] =      { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Sin },
    [cos_op] =      { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Cos },

    [abs_op] =  { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = { (SpvOp) GLSLstd450SAbs,  ISEL_ILLEGAL, (SpvOp) GLSLstd450FAbs,  ISEL_ILLEGAL }},
    [sign_op] = { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = { (SpvOp) GLSLstd450SSign, ISEL_ILLEGAL, (SpvOp) GLSLstd450FSign, ISEL_ILLEGAL }},

    [min_op] = { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMin, (SpvOp) GLSLstd450UMin, (SpvOp) GLSLstd450FMin, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [max_op] = { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMax, (SpvOp) GLSLstd450UMax, (SpvOp) GLSLstd450FMax, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [exp_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Exp },
    [pow_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Pow },
    [fma_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Fma },

    [sample_texture_op] = {Plain, Monomorphic, TyOperand, .op = SpvOpImageSampleImplicitLod },

    [subgroup_assume_uniform_op] = {Plain, Monomorphic, Same, .op = ISEL_IDENTITY },

    [PRIMOPS_COUNT] = { Custom }
};

#pragma GCC diagnostic pop
#pragma GCC diagnostic error "-Wswitch"

static const Type* get_result_t(Emitter* emitter, IselTableEntry entry, Nodes args, Nodes type_arguments) {
    switch (entry.result_kind) {
        case Same:      return get_unqualified_type(first(args)->type);
        case SameTuple: return record_type(emitter->arena, (RecordType) { .members = mk_nodes(emitter->arena, get_unqualified_type(first(args)->type), get_unqualified_type(first(args)->type)) });
        case Bool:      return bool_type(emitter->arena);
        case TyOperand: return first(type_arguments);
        case Void:      return unit_type(emitter->arena);
    }
}

static SpvOp get_opcode(SHADY_UNUSED Emitter* emitter, IselTableEntry entry, Nodes args, Nodes type_arguments) {
    switch (entry.isel_mechanism) {
        case None:        return SpvOpMax;
        case Monomorphic: return entry.op;
        case FirstOp: {
            assert(args.count >= 1);
            OperandClass op_class = classify_operand_type(get_unqualified_type(first(args)->type));
            return entry.fo[op_class];
        }
        case FirstAndResult: {
            assert(args.count >= 1);
            assert(type_arguments.count == 1);
            OperandClass op_class = classify_operand_type(get_unqualified_type(first(args)->type));
            OperandClass return_t_class = classify_operand_type(first(type_arguments));
            return entry.foar[op_class][return_t_class];
        }
    }
}

static SpvId emit_primop(Emitter* emitter, BBBuilder bb_builder, const Node* instr) {
    PrimOp the_op = instr->payload.prim_op;
    Nodes args = the_op.operands;
    Nodes type_arguments = the_op.type_arguments;

    IselTableEntry entry = isel_table[the_op.op];
    if (entry.class != Custom) {
        LARRAY(SpvId, emitted_args, args.count);
        for (size_t i = 0; i < args.count; i++)
            emitted_args[i] = spv_emit_value(emitter, args.nodes[i]);

        switch (entry.class) {
            case Plain: {
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                if (opcode == SpvOpNop) {
                    assert(args.count == 1);
                    return emitted_args[0];
                }

                if (opcode == SpvOpMax)
                    goto custom;

                SpvId result_t = instr->type == empty_multiple_return_type(emitter->arena) ? emit_type(emitter, instr->type) : emitter->void_t;
                if (entry.extended_set) {
                    SpvId set_id = get_extended_instruction_set(emitter, entry.extended_set);
                    return spvb_ext_instruction(bb_builder, result_t, set_id, opcode, args.count, emitted_args);
                } else {
                    return spvb_op(bb_builder, opcode, result_t, args.count, emitted_args);
                }
            }
            case Custom: SHADY_UNREACHABLE;
        }
        SHADY_UNREACHABLE;
    }

    custom:
    switch (the_op.op) {
        case reinterpret_op: {
            const Type* dst = first(the_op.type_arguments);
            const Type* src = get_unqualified_type(first(the_op.operands)->type);
            assert(dst->tag == PtrType_TAG && src->tag == PtrType_TAG);
            assert(src != dst);
            return spvb_op(bb_builder, SpvOpBitcast, emit_type(emitter, dst), 1, (SpvId[]) {spv_emit_value(emitter, fn_builder, first(the_op.operands)) });
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            bool insert = the_op.op == insert_op;

            const Node* src_value = first(args);
            const Type* result_t = instr->type;
            size_t indices_start = insert ? 2 : 1;
            size_t indices_count = args.count - indices_start;
            assert(args.count > indices_start);

            bool dynamic = the_op.op == extract_dynamic_op;

            if (dynamic) {
                LARRAY(SpvId, indices, indices_count);
                for (size_t i = 0; i < indices_count; i++) {
                    indices[i] = spv_emit_value(emitter, args.nodes[i + indices_start]);
                }
                assert(indices_count == 1);
                return spvb_vector_extract_dynamic(bb_builder, emit_type(emitter, result_t), spv_emit_value(emitter, src_value), indices[0]);
            }
            LARRAY(uint32_t, indices, indices_count);
            for (size_t i = 0; i < indices_count; i++) {
                // TODO: fallback to Dynamic variants transparently
                indices[i] = get_int_literal_value(*resolve_to_int_literal(args.nodes[i + indices_start]), false);
            }

            if (insert)
                return spvb_insert(bb_builder, emit_type(emitter, result_t), spv_emit_value(emitter, args.nodes[1]), spv_emit_value(emitter, src_value), indices_count, indices);
            else
                return spvb_extract(bb_builder, emit_type(emitter, result_t), spv_emit_value(emitter, src_value), indices_count, indices);
        }
        case shuffle_op: {
            const Type* result_t = instr->type;
            SpvId a = spv_emit_value(emitter, args.nodes[0]);
            SpvId b = spv_emit_value(emitter, args.nodes[1]);
            LARRAY(uint32_t, indices, args.count - 2);
            for (size_t i = 0; i < args.count - 2; i++) {
                int64_t indice = get_int_literal_value(*resolve_to_int_literal(args.nodes[i + 2]), true);
                if (indice == -1)
                    indices[i] = 0xFFFFFFFF;
                else
                    indices[i] = indice;
            }
            return spvb_vecshuffle(bb_builder, emit_type(emitter, result_t), a, b, args.count - 2, indices);
        }
        case select_op: {
            SpvId cond = spv_emit_value(emitter, first(args));
            SpvId truv = spv_emit_value(emitter, args.nodes[1]);
            SpvId flsv = spv_emit_value(emitter, args.nodes[2]);

            return spvb_select(bb_builder, emit_type(emitter, args.nodes[1]->type), cond, truv, flsv);
        }
        default: error("TODO: unhandled op");
    }
    error("unreachable");
}

static SpvId emit_ext_instr(Emitter* emitter, BBBuilder bb_builder, ExtInstr instr) {
    if (strcmp("spirv.core", instr.set) == 0) {
        switch (instr.opcode) {
            case SpvOpGroupNonUniformBroadcastFirst: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
                SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
                if (emitter->configuration->hacks.spv_shuffle_instead_of_broadcast_first) {
                    spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformShuffle);
                    const Node* b = ref_decl_helper(emitter->arena, get_or_create_builtin(emitter->module, BuiltinSubgroupLocalInvocationId, NULL));
                    SpvId local_id = spvb_op(bb_builder, SpvOpLoad, emit_type(emitter, uint32_type(emitter->arena)), 1, (SpvId []) { spv_emit_value(emitter, fn_builder, b) });
                    return spvb_group_shuffle(bb_builder, emit_type(emitter, instr.result_t), scope_subgroup, spv_emit_value(emitter, fn_builder, first(instr.operands)), local_id);
                }
                break;
            }
            case SpvCapabilityGroupNonUniformBallot: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
                assert(instr.operands.count == 2);
                // SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
                // ad-hoc extension for my sanity
                if (get_unqualified_type(instr.result_t) == get_actual_mask_type(emitter->arena)) {
                    const Type* i32x4 = pack_type(emitter->arena, (PackType) { .width = 4, .element_type = uint32_type(emitter->arena) });
                    SpvId raw_result = spvb_group_ballot(bb_builder, emit_type(emitter, i32x4), spv_emit_value(emitter, fn_builder, instr.operands.nodes[1]), spv_emit_value(emitter, fn_builder, first(instr.operands)));
                    // TODO: why are we doing this in SPIR-V and not the IR ?
                    SpvId low32 = spvb_extract(bb_builder, emit_type(emitter, uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 0 });
                    SpvId hi32 = spvb_extract(bb_builder, emit_type(emitter, uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 1 });
                    SpvId low64 = spvb_op(bb_builder, SpvOpUConvert, emit_type(emitter, uint64_type(emitter->arena)), 1, &low32);
                    SpvId hi64 = spvb_op(bb_builder, SpvOpUConvert, emit_type(emitter, uint64_type(emitter->arena)), 1, &hi32);
                    hi64 = spvb_op(bb_builder, SpvOpShiftLeftLogical, emit_type(emitter, uint64_type(emitter->arena)), 2, (SpvId []) { hi64, spv_emit_value(emitter, fn_builder, int64_literal(emitter->arena, 32)) });
                    SpvId final_result = spvb_op(bb_builder, SpvOpBitwiseOr, emit_type(emitter, uint64_type(emitter->arena)), 2, (SpvId []) { low64, hi64 });
                    return final_result;
                }
                break;
            }
            case SpvOpGroupNonUniformIAdd: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformArithmetic);
                break;
                // SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
                // return spvb_group_non_uniform_iadd(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), spv_emit_value(emitter, fn_builder, first(args)), scope_subgroup, SpvGroupOperationReduce, NULL);
            }
            case SpvOpGroupNonUniformElect: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniform);
                assert(instr.operands.count == 1);
                break;
                // SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
                // SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
                // return spvb_group_elect(bb_builder, result_t, scope_subgroup);
            }
            default: break;
        }
        LARRAY(SpvId, ops, instr.operands.count);
        for (size_t i = 0; i < instr.operands.count; i++)
            ops[i] = spv_emit_value(emitter, instr.operands.nodes[i]);
        return spvb_op(bb_builder, instr.opcode, emit_type(emitter, instr.result_t), instr.operands.count, ops);
    }
    LARRAY(SpvId, ops, instr.operands.count);
    for (size_t i = 0; i < instr.operands.count; i++)
        ops[i] = spv_emit_value(emitter, instr.operands.nodes[i]);
    SpvId set_id = get_extended_instruction_set(emitter, instr.set);
    return spvb_ext_instruction(bb_builder, emit_type(emitter, instr.result_t), set_id, instr.opcode, instr.operands.count, ops);
}

static SpvId emit_leaf_call(Emitter* emitter, BBBuilder bb_builder, Call call) {
    const Node* fn = call.callee;
    assert(fn->tag == FnAddr_TAG);
    fn = fn->payload.fn_addr.fn;
    SpvId callee = emit_decl(emitter, fn);

    const Type* callee_type = fn->type;
    assert(callee_type->tag == FnType_TAG);
    Nodes return_types = callee_type->payload.fn_type.return_types;
    SpvId return_type = nodes_to_codom(emitter, return_types);
    LARRAY(SpvId, args, call.args.count);
    for (size_t i = 0; i < call.args.count; i++)
        args[i] = spv_emit_value(emitter, call.args.nodes[i]);
    return spvb_call(bb_builder, return_type, callee, call.args.count, args);
}

static SpvId spv_emit_instruction(Emitter* emitter, BBBuilder bb_builder, const Node* instruction) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: error("");
        case Instruction_PushStack_TAG:
        case Instruction_PopStack_TAG:
        case Instruction_GetStackSize_TAG:
        case Instruction_SetStackSize_TAG:
        case Instruction_GetStackBaseAddr_TAG: error("Stack operations need to be lowered.");
        case Instruction_CopyBytes_TAG:
        case Instruction_FillBytes_TAG:
        case Instruction_BindIdentifiers_TAG:
        case Instruction_StackAlloc_TAG: error("Should be lowered elsewhere")
        case Instruction_ExtInstr_TAG: return emit_ext_instr(emitter, bb_builder, instruction->payload.ext_instr);
        case Instruction_Call_TAG: return emit_leaf_call(emitter, bb_builder, instruction->payload.call);
        case PrimOp_TAG: return emit_primop(emitter, bb_builder, instruction);
        case Comment_TAG: break;
        case Instruction_LocalAlloc_TAG: {
            assert(bb_builder);
            return spvb_local_variable(spvb_get_fn_builder(bb_builder), emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunction,
                .pointed_type = instruction->payload.local_alloc.type
            })), SpvStorageClassFunction);
        }
        case Instruction_Load_TAG: {
            Load payload = instruction->payload.load;
            const Type* ptr_type = payload.ptr->type;
            deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobal) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = spv_emit_value(emitter, payload.ptr);
            return spvb_load(bb_builder, emit_type(emitter, elem_type), eptr, operands_count, operands);
        }
        case Instruction_Store_TAG: {
            Store payload = instruction->payload.store;
            const Type* ptr_type = payload.ptr->type;
            deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobal) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = spv_emit_value(emitter, payload.ptr);
            SpvId eval = spv_emit_value(emitter, payload.value);
            spvb_store(bb_builder, eval, eptr, operands_count, operands);
            return 0;
        }
        case Lea_TAG: {
            Lea payload = instruction->payload.lea;
            SpvId base = spv_emit_value(emitter, payload.ptr);

            LARRAY(SpvId, indices, payload.indices.count);
            for (size_t i = 0; i < payload.indices.count; i++)
                indices[i] = payload.indices.nodes[i] ? spv_emit_value(emitter, payload.indices.nodes[i]) : 0;

            const IntLiteral* known_offset = resolve_to_int_literal(payload.offset);
            if (known_offset && known_offset->value == 0) {
                const Type* target_type = instruction->type;
                return spvb_access_chain(bb_builder, emit_type(emitter, target_type), base, payload.indices.count, indices);
            } else {
                const Type* target_type = instruction->type;
                return spvb_ptr_access_chain(bb_builder, emit_type(emitter, target_type), base, spv_emit_value(emitter, payload.offset), payload.indices.count, indices);
            }
        }
        case Instruction_DebugPrintf_TAG: {
            SpvId set_id = get_extended_instruction_set(emitter, "NonSemantic.DebugPrintf");
            LARRAY(SpvId, args, instruction->payload.debug_printf.args.count + 1);
            args[0] = spv_emit_value(emitter, string_lit_helper(emitter->arena, instruction->payload.debug_printf.string));
            for (size_t i = 0; i < instruction->payload.debug_printf.args.count; i++)
                args[i + 1] = spv_emit_value(emitter, instruction->payload.debug_printf.args.nodes[i]);
            spvb_ext_instruction(bb_builder, emit_type(emitter, instruction->type), set_id, (SpvOp) NonSemanticDebugPrintfDebugPrintf, instruction->payload.debug_printf.args.count + 1, args);
            return 0;
        }
    }
}

static SpvId spv_emit_value_(Emitter* emitter, BBBuilder bb_builder, const Node* node) {
    if (is_instruction(node))
        return spv_emit_instruction(emitter, bb_builder, node);

    SpvId new;
    switch (is_value(node)) {
        case NotAValue: error("");
        case Param_TAG: error("tried to emit a param: all params should be emitted by their binding abstraction !");
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
                ids[i] = spv_emit_value(emitter, contents.nodes[i]);
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
                    new = spv_emit_value(emitter, decl->payload.constant.value);
                    break;
                }
                default: error("RefDecl must reference a constant or global");
            }
            break;
        }
        default: {
            error("Unhandled value for code generation: %s", node_tags[node->tag]);
        }
    }

    return new;
}

SpvId spv_emit_value(Emitter* emitter, const Node* node) {
    SpvId* existing = spv_search_emitted(emitter, node);
    if (existing)
        return *existing;

    CFNode* where = emitter->scheduler ? schedule_instruction(emitter->scheduler, node) : NULL;
    if (where) {
        BBBuilder bb_builder = spv_find_basic_block_builder(emitter, where->node);
        SpvId emitted = spv_emit_value_(emitter, bb_builder, node);
        register_result(emitter, false, node, emitted);
        return emitted;
    } else {
        SpvId emitted = spv_emit_value_(emitter, NULL, node);
        register_result(emitter, true, node, emitted);
        return emitted;
    }
}
