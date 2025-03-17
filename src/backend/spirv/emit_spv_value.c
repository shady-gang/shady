#include "emit_spv.h"

#include "shady/ir/memory_layout.h"
#include "shady/ir/builtin.h"

#include "../shady/analysis/cfg.h"
#include "../shady/analysis/scheduler.h"

#include "log.h"
#include "dict.h"
#include "portability.h"

#include "spirv/unified1/NonSemanticDebugPrintf.h"
#include "spirv/unified1/GLSL.std.450.h"

#include <assert.h>
#include <string.h>

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
    assert(is_type(type) && shd_is_data_type(type));

    if (type->tag == PackType_TAG)
        return classify_operand_type(type->payload.pack_type.element_type);

    switch (type->tag) {
        case Int_TAG:     return type->payload.int_type.is_signed ? Signed : Unsigned;
        case Bool_TAG:    return Logical;
        case PtrType_TAG: return Ptr;
        case Float_TAG:   return FP;
        default: shd_error("we don't know what to do with this")
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

    [PRIMOPS_COUNT] = { Custom }
};

#pragma GCC diagnostic pop
#pragma GCC diagnostic error "-Wswitch"

static const Type* get_result_t(Emitter* emitter, IselTableEntry entry, Nodes args, Nodes type_arguments) {
    switch (entry.result_kind) {
        case Same:      return shd_get_unqualified_type(shd_first(args)->type);
        case SameTuple: return record_type(emitter->arena, (RecordType) { .members = mk_nodes(emitter->arena, shd_get_unqualified_type(shd_first(args)->type), shd_get_unqualified_type(shd_first(args)->type)) });
        case Bool:      return bool_type(emitter->arena);
        case TyOperand: return shd_first(type_arguments);
        case Void:      return unit_type(emitter->arena);
    }
}

static SpvOp get_opcode(SHADY_UNUSED Emitter* emitter, IselTableEntry entry, Nodes args, Nodes type_arguments) {
    switch (entry.isel_mechanism) {
        case None:        return SpvOpMax;
        case Monomorphic: return entry.op;
        case FirstOp: {
            assert(args.count >= 1);
            OperandClass op_class = classify_operand_type(shd_get_unqualified_type(shd_first(args)->type));
            return entry.fo[op_class];
        }
        case FirstAndResult: {
            assert(args.count >= 1);
            assert(type_arguments.count == 1);
            OperandClass op_class = classify_operand_type(shd_get_unqualified_type(shd_first(args)->type));
            OperandClass return_t_class = classify_operand_type(shd_first(type_arguments));
            return entry.foar[op_class][return_t_class];
        }
    }
}

static SpvId emit_primop(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* instr) {
    PrimOp the_op = instr->payload.prim_op;
    Nodes args = the_op.operands;
    Nodes type_arguments = the_op.type_arguments;

    IselTableEntry entry = isel_table[the_op.op];
    if (entry.class != Custom) {
        LARRAY(SpvId, emitted_args, args.count);
        for (size_t i = 0; i < args.count; i++)
            emitted_args[i] = spv_emit_value(emitter, fn_builder, args.nodes[i]);

        switch (entry.class) {
            case Plain: {
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                if (opcode == SpvOpNop) {
                    assert(args.count == 1);
                    return emitted_args[0];
                }

                if (opcode == SpvOpMax)
                    goto custom;

                SpvId result_t = instr->type == empty_multiple_return_type(emitter->arena) ? emitter->void_t : spv_emit_type(emitter, instr->type);
                if (entry.extended_set) {
                    SpvId set_id = spv_get_extended_instruction_set(emitter, entry.extended_set);
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
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            bool insert = the_op.op == insert_op;

            const Node* src_value = shd_first(args);
            const Type* result_t = instr->type;
            size_t indices_start = insert ? 2 : 1;
            size_t indices_count = args.count - indices_start;
            assert(args.count > indices_start);

            bool dynamic = the_op.op == extract_dynamic_op;

            if (dynamic) {
                LARRAY(SpvId, indices, indices_count);
                for (size_t i = 0; i < indices_count; i++) {
                    indices[i] = spv_emit_value(emitter, fn_builder, args.nodes[i + indices_start]);
                }
                assert(indices_count == 1);
                return spvb_vector_extract_dynamic(bb_builder, spv_emit_type(emitter, result_t), spv_emit_value(emitter, fn_builder, src_value), indices[0]);
            }
            LARRAY(uint32_t, indices, indices_count);
            for (size_t i = 0; i < indices_count; i++) {
                // TODO: fallback to Dynamic variants transparently
                indices[i] = shd_get_int_literal_value(*shd_resolve_to_int_literal(args.nodes[i + indices_start]), false);
            }

            if (insert)
                return spvb_insert(bb_builder, spv_emit_type(emitter, result_t), spv_emit_value(emitter, fn_builder, args.nodes[1]), spv_emit_value(emitter, fn_builder, src_value), indices_count, indices);
            else
                return spvb_extract(bb_builder, spv_emit_type(emitter, result_t), spv_emit_value(emitter, fn_builder, src_value), indices_count, indices);
        }
        case shuffle_op: {
            const Type* result_t = instr->type;
            SpvId a = spv_emit_value(emitter, fn_builder, args.nodes[0]);
            SpvId b = spv_emit_value(emitter, fn_builder, args.nodes[1]);
            LARRAY(uint32_t, indices, args.count - 2);
            for (size_t i = 0; i < args.count - 2; i++) {
                int64_t indice = shd_get_int_literal_value(*shd_resolve_to_int_literal(args.nodes[i + 2]), true);
                if (indice == -1)
                    indices[i] = 0xFFFFFFFF;
                else
                    indices[i] = indice;
            }
            return spvb_vecshuffle(bb_builder, spv_emit_type(emitter, result_t), a, b, args.count - 2, indices);
        }
        case select_op: {
            SpvId cond = spv_emit_value(emitter, fn_builder, shd_first(args));
            SpvId truv = spv_emit_value(emitter, fn_builder, args.nodes[1]);
            SpvId flsv = spv_emit_value(emitter, fn_builder, args.nodes[2]);

            return spvb_select(bb_builder, spv_emit_type(emitter, args.nodes[1]->type), cond, truv, flsv);
        }
        default: shd_error("TODO: unhandled op");
    }
    shd_error("unreachable");
}

static SpvId emit_ext_op(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Type* result_t, String set, SpvOp opcode, Nodes operands) {
    if (strcmp("spirv.core", set) == 0) {
        switch (opcode) {
            case SpvOpGroupNonUniformBroadcastFirst: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
                SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, shd_int32_literal(emitter->arena, SpvScopeSubgroup));
                if (emitter->spirv_tgt.hacks.shuffle_instead_of_broadcast_first) {
                    spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformShuffle);
                    const Node* b = shd_get_or_create_builtin(emitter->module, BuiltinSubgroupLocalInvocationId);
                    SpvId local_id = spvb_op(bb_builder, SpvOpLoad, spv_emit_type(emitter, shd_uint32_type(emitter->arena)), 1, (SpvId []) { spv_emit_value(emitter, fn_builder, b) });
                    return spvb_group_shuffle(bb_builder, spv_emit_type(emitter, result_t), scope_subgroup, spv_emit_value(emitter, fn_builder, shd_first(operands)), local_id);
                }
                break;
            }
            case SpvOpGroupNonUniformBallot: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
                assert(operands.count == 2);
                // SpvId scope_subgroup = spv_emit_value(emitter, fn_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
                // ad-hoc extension for my sanity
                if (shd_get_unqualified_type(result_t) == shd_get_actual_mask_type(emitter->arena)) {
                    const Type* i32x4 = pack_type(emitter->arena, (PackType) { .width = 4, .element_type = shd_uint32_type(emitter->arena) });
                    SpvId raw_result = spvb_group_ballot(bb_builder, spv_emit_type(emitter, i32x4), spv_emit_value(emitter, fn_builder, operands.nodes[1]), spv_emit_value(emitter, fn_builder, shd_first(operands)));
                    // TODO: why are we doing this in SPIR-V and not the IR ?
                    SpvId low32 = spvb_extract(bb_builder, spv_emit_type(emitter, shd_uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 0 });
                    SpvId hi32 = spvb_extract(bb_builder, spv_emit_type(emitter, shd_uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 1 });
                    SpvId low64 = spvb_op(bb_builder, SpvOpUConvert, spv_emit_type(emitter, shd_uint64_type(emitter->arena)), 1, &low32);
                    SpvId hi64 = spvb_op(bb_builder, SpvOpUConvert, spv_emit_type(emitter, shd_uint64_type(emitter->arena)), 1, &hi32);
                    hi64 = spvb_op(bb_builder, SpvOpShiftLeftLogical, spv_emit_type(emitter, shd_uint64_type(emitter->arena)), 2, (SpvId []) { hi64, spv_emit_value(emitter, fn_builder, shd_int64_literal(emitter->arena, 32)) });
                    SpvId final_result = spvb_op(bb_builder, SpvOpBitwiseOr, spv_emit_type(emitter, shd_uint64_type(emitter->arena)), 2, (SpvId []) { low64, hi64 });
                    return final_result;
                }
                break;
            }
            case SpvOpGroupNonUniformIAdd: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformArithmetic);
                SpvId scope = spv_emit_value(emitter, fn_builder, shd_first(operands));
                SpvGroupOperation group_op = shd_get_int_literal_value(*shd_resolve_to_int_literal(operands.nodes[1]), false);
                return spvb_group_non_uniform_group_op(bb_builder, spv_emit_type(emitter, result_t), opcode, scope, group_op, spv_emit_value(emitter, fn_builder, operands.nodes[2]), NULL);
            }
            case SpvOpGroupNonUniformElect: {
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniform);
                assert(operands.count == 1);
                break;
            }
            default: break;
        }
        LARRAY(SpvId, ops, operands.count);
        for (size_t i = 0; i < operands.count; i++)
            ops[i] = spv_emit_value(emitter, fn_builder, operands.nodes[i]);
        return spvb_op(bb_builder, opcode, spv_emit_type(emitter, result_t), operands.count, ops);
    }
    LARRAY(SpvId, ops, operands.count);
    for (size_t i = 0; i < operands.count; i++)
        ops[i] = spv_emit_value(emitter, fn_builder, operands.nodes[i]);
    SpvId set_id = spv_get_extended_instruction_set(emitter, set);
    return spvb_ext_instruction(bb_builder, spv_emit_type(emitter, result_t), set_id, opcode, operands.count, ops);
}

static SpvId emit_fn_call(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* callee, Nodes args, const Type* return_type) {
    LARRAY(SpvId, eargs, args.count);
    for (size_t i = 0; i < args.count; i++)
        eargs[i] = spv_emit_value(emitter, fn_builder, args.nodes[i]);

    if (callee->tag == Function_TAG) {
        return spvb_call(bb_builder, spv_emit_type(emitter, return_type), spv_emit_decl(emitter, callee), args.count, eargs);
    } else {
        spvb_capability(emitter->file_builder, SpvCapabilityFunctionPointersINTEL);
        return spvb_op(bb_builder, SpvOpFunctionPointerCallINTEL, spv_emit_type(emitter, return_type), args.count, eargs);
    }
}

static SpvId spv_emit_instruction(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* instruction) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: shd_error("");
        case Instruction_PushStack_TAG:
        case Instruction_PopStack_TAG:
        case Instruction_GetStackSize_TAG:
        case Instruction_SetStackSize_TAG:
        case Instruction_GetStackBaseAddr_TAG: shd_error("Stack operations need to be lowered.");
        case Instruction_CopyBytes_TAG:
        case Instruction_FillBytes_TAG:
        case Instruction_StackAlloc_TAG: shd_error("Should be lowered elsewhere")
        case Instruction_ExtInstr_TAG: {
            ExtInstr instr = instruction->payload.ext_instr;
            spv_emit_mem(emitter, fn_builder, instr.mem);
            return emit_ext_op(emitter, fn_builder, bb_builder, instr.result_t, instr.set, instr.opcode, instr.operands);
        }
        case Instruction_Call_TAG: {
            Call payload = instruction->payload.call;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            return emit_fn_call(emitter, fn_builder, bb_builder, payload.callee, payload.args, instruction->type);
        } case Instruction_IndirectCall_TAG: {
            IndirectCall payload = instruction->payload.indirect_call;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            return emit_fn_call(emitter, fn_builder, bb_builder, payload.callee, payload.args, instruction->type);
        } case PrimOp_TAG: return emit_primop(emitter, fn_builder, bb_builder, instruction);
        case Comment_TAG: {
            spv_emit_mem(emitter, fn_builder, instruction->payload.comment.mem);
            return 0;
        }
        case Instruction_LocalAlloc_TAG: {
            LocalAlloc payload = instruction->payload.local_alloc;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            assert(bb_builder);
            SpvId id = spvb_local_variable(spvb_get_fn_builder(bb_builder), spv_emit_type(emitter, instruction->type), SpvStorageClassFunction);
            spv_emit_aliased_restrict(emitter, id, instruction->type);
            return id;
        }
        case Instruction_Load_TAG: {
            Load payload = instruction->payload.load;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Type* ptr_type = payload.ptr->type;
            shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobal) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = shd_get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = spv_emit_value(emitter, fn_builder, payload.ptr);
            return spvb_load(bb_builder, spv_emit_type(emitter, elem_type), eptr, operands_count, operands);
        }
        case Instruction_Store_TAG: {
            Store payload = instruction->payload.store;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            const Type* ptr_type = payload.ptr->type;
            shd_deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobal) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = shd_get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = spv_emit_value(emitter, fn_builder, payload.ptr);
            SpvId eval = spv_emit_value(emitter, fn_builder, payload.value);
            spvb_store(bb_builder, eval, eptr, operands_count, operands);
            return 0;
        }
        case Instruction_BitCast_TAG: {
            BitCast payload = instruction->payload.bit_cast;
            bool src_ptr = shd_get_unqualified_type(payload.src->type)->tag == PtrType_TAG;
            bool dst_ptr = payload.type->tag == PtrType_TAG;
            SpvOp op = SpvOpBitcast;
            if (src_ptr && !dst_ptr)
                op = SpvOpConvertPtrToU;
            else if (!src_ptr && dst_ptr)
                op = SpvOpConvertUToPtr;
            SpvId src = spv_emit_value(emitter, fn_builder, payload.src);
            return spvb_op(bb_builder, op, spv_emit_type(emitter, instruction->type), 1, &src);
        }
        case Instruction_ScopeCast_TAG: {
            SpvId new = spv_emit_value(emitter, fn_builder, instruction->payload.scope_cast.src);
            new = spvb_op(bb_builder, SpvOpCopyObject, spv_emit_type(emitter, instruction->type), 1, &new);
            return new;
        }
        case Instruction_PtrCompositeElement_TAG: {
            PtrCompositeElement payload = instruction->payload.ptr_composite_element;
            SpvId base = spv_emit_value(emitter, fn_builder, payload.ptr);
            const Type* target_type = instruction->type;
            SpvId index = spv_emit_value(emitter, fn_builder, payload.index);
            return spvb_access_chain(bb_builder, spv_emit_type(emitter, target_type), base, 1, &index);
        }
        case Instruction_PtrArrayElementOffset_TAG: {
            PtrArrayElementOffset payload = instruction->payload.ptr_array_element_offset;
            SpvId base = spv_emit_value(emitter, fn_builder, payload.ptr);
            const Type* target_type = instruction->type;
            SpvId offset = spv_emit_value(emitter, fn_builder, payload.offset);
            return spvb_ptr_access_chain(bb_builder, spv_emit_type(emitter, target_type), base, offset, 0, NULL);
        }
        case Instruction_DebugPrintf_TAG: {
            DebugPrintf payload = instruction->payload.debug_printf;
            spv_emit_mem(emitter, fn_builder, payload.mem);
            SpvId set_id = spv_get_extended_instruction_set(emitter, "NonSemantic.DebugPrintf");
            LARRAY(SpvId, args, instruction->payload.debug_printf.args.count + 1);
            args[0] = spv_emit_value(emitter, fn_builder, string_lit_helper(emitter->arena, instruction->payload.debug_printf.string));
            for (size_t i = 0; i < instruction->payload.debug_printf.args.count; i++)
                args[i + 1] = spv_emit_value(emitter, fn_builder, instruction->payload.debug_printf.args.nodes[i]);
            spvb_ext_instruction(bb_builder, spv_emit_type(emitter, instruction->type), set_id, (SpvOp) NonSemanticDebugPrintfDebugPrintf, instruction->payload.debug_printf.args.count + 1, args);
            return 0;
        }
    }
    SHADY_UNREACHABLE;
}

static SpvId spv_emit_value_(Emitter* emitter, FnBuilder* fn_builder, BBBuilder bb_builder, const Node* node) {
    if (is_instruction(node))
        return spv_emit_instruction(emitter, fn_builder, bb_builder, node);

    SpvId new;
    switch (is_value(node)) {
        case NotAValue: shd_error("");
        case Param_TAG: shd_error("tried to emit a param: all params should be emitted by their binding abstraction !");
        case Value_UntypedNumber_TAG:
        case Value_FnAddr_TAG: {
            spvb_capability(emitter->file_builder, SpvCapabilityInModuleFunctionAddressSHADY);
            SpvId fn = spv_emit_decl(emitter, node->payload.fn_addr.fn);
            return spvb_constant_op(emitter->file_builder, spv_emit_type(emitter, node->type), SpvOpConstantFunctionAddressSHADY, 1, &fn);
        }
        case IntLiteral_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            SpvId ty = spv_emit_type(emitter, node->type);
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
            SpvId ty = spv_emit_type(emitter, node->type);
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
            spvb_bool_constant(emitter->file_builder, new, spv_emit_type(emitter, bool_type(emitter->arena)), true);
            break;
        }
        case False_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            spvb_bool_constant(emitter->file_builder, new, spv_emit_type(emitter, bool_type(emitter->arena)), false);
            break;
        }
        case Value_StringLiteral_TAG: {
            new = spvb_debug_string(emitter->file_builder, node->payload.string_lit.string);
            break;
        }
        case Value_NullPtr_TAG: {
            new = spvb_constant_null(emitter->file_builder, spv_emit_type(emitter, node->payload.null_ptr.ptr_type));
            break;
        }
        case Composite_TAG: {
            Nodes contents = node->payload.composite.contents;
            LARRAY(SpvId, ids, contents.count);
            for (size_t i = 0; i < contents.count; i++) {
                ids[i] = spv_emit_value(emitter, fn_builder, contents.nodes[i]);
            }
            if (bb_builder) {
                new = spvb_composite(bb_builder, spv_emit_type(emitter, node->type), contents.count, ids);
                return new;
            } else {
                new = spvb_constant_composite(emitter->file_builder, spv_emit_type(emitter, node->type), contents.count, ids);
                break;
            }
        }
        case Value_Undef_TAG: {
            new = spvb_undef(emitter->file_builder, spv_emit_type(emitter, node->payload.undef.type));
            break;
        }
        case Value_Fill_TAG: shd_error("lower me")
        case Value_GlobalVariable_TAG:
        case Value_Constant_TAG: return spv_emit_decl(emitter, node);
        case Value_BuiltinRef_TAG: {
            BuiltinRef payload = node->payload.builtin_ref;
            SpvId given_id = spvb_fresh_id(emitter->file_builder);
            AddressSpace as = shd_get_builtin_address_space(payload.builtin);
            SpvStorageClass storage_class = spv_emit_addr_space(emitter, as);
            spvb_global_variable(emitter->file_builder, given_id, spv_emit_type(emitter, node->type), storage_class, false, 0);

            SpvBuiltIn d = shd_get_builtin_spv_id(payload.builtin);
            uint32_t decoration_payload[] = { d };
            spvb_decorate(emitter->file_builder, given_id, SpvDecorationBuiltIn, 1, decoration_payload);
            if (as == AsUInput || as == AsInput) {
                const Type* element_type = shd_get_builtin_type(emitter->arena, payload.builtin);
                if (element_type->tag == Int_TAG)
                    spvb_decorate(emitter->file_builder, given_id, SpvDecorationFlat, 0, NULL);
            }
            shd_spv_register_interface(emitter, node, given_id);
            return given_id;
        }
        case ExtValue_TAG: {
            ExtValue instr = node->payload.ext_value;
            return emit_ext_op(emitter, fn_builder, bb_builder, instr.result_t, instr.set, instr.opcode, instr.operands);
        }
        default: {
            shd_error("Unhandled value for code generation: %s", shd_get_node_tag_string(node->tag));
        }
    }

    return new;
}

static bool can_appear_at_top_level(const Node* node) {
    switch (node->tag) {
        case Undef_TAG:
        case FloatLiteral_TAG:
        case IntLiteral_TAG:
        case True_TAG:
        case False_TAG:
        case BuiltinRef_TAG:
            return true;
        case ScopeCast_TAG:
            return can_appear_at_top_level(node->payload.scope_cast.src);
        case Composite_TAG: {
            bool ok = true;
            Nodes components = node->payload.composite.contents;
            for (size_t i = 0; i < components.count; i++) {
                ok &= can_appear_at_top_level(components.nodes[i]);
            }
            return ok;
        }
        default: break;
    }
    return false;
}

SpvId spv_emit_value(Emitter* emitter, FnBuilder* fn_builder, const Node* node) {
    SpvId* existing = spv_search_emitted(emitter, fn_builder, node);
    if (existing)
        return *existing;

    CFNode* where = fn_builder ? shd_schedule_instruction(fn_builder->scheduler, node) : NULL;
    if (where) {
        BBBuilder bb_builder = spv_find_basic_block_builder(emitter, where->node);
        SpvId emitted = spv_emit_value_(emitter, fn_builder, bb_builder, node);
        spv_register_emitted(emitter, fn_builder, node, emitted);
        return emitted;
    } else if (!can_appear_at_top_level(node)) {
        if (!fn_builder) {
            shd_log_node(ERROR, node);
            shd_log_fmt(ERROR, "cannot appear at top-level");
            exit(-1);
        }
        // Pick the entry block of the current fn
        BBBuilder bb_builder = spv_find_basic_block_builder(emitter, fn_builder->cfg->entry->node);
        SpvId emitted = spv_emit_value_(emitter, fn_builder, bb_builder, node);
        spv_register_emitted(emitter, fn_builder, node, emitted);
        return emitted;
    } else {
        assert(!is_mem(node));
        SpvId emitted = spv_emit_value_(emitter, NULL, NULL, node);
        spv_register_emitted(emitter, NULL, node, emitted);
        return emitted;
    }
}

SpvId spv_emit_mem(Emitter* e, FnBuilder* b, const Node* mem) {
    assert(is_mem(mem));
    if (mem->tag == AbsMem_TAG)
        return 0;
    if (is_instruction(mem))
        return spv_emit_value(e, b, mem);
    shd_error("What sort of mem is this ?");
}

/// Dumb nonsense: this isn't consistently emitted for pointers in complex data structures
/// These qualifiers should, at most, go on SSA _values_, not on variables !
/// but the validator wants it so, oh well...
void spv_emit_aliased_restrict(Emitter* emitter, SpvId id, const Type* t) {
    assert(t->tag == QualifiedType_TAG);
    t = shd_get_unqualified_type(t);
    if (t->tag == PtrType_TAG) {
        PtrType payload = t->payload.ptr_type;
        if (payload.pointed_type->tag == PtrType_TAG && payload.pointed_type->payload.ptr_type.address_space == AsGlobal)
            spvb_decorate(emitter->file_builder, id, SpvDecorationAliasedPointer, 0, NULL);
    }
}
