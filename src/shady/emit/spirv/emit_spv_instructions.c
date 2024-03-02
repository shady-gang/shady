#include "emit_spv.h"

#include "log.h"
#include "portability.h"

#include "../../type.h"
#include "../../transform/memory_layout.h"
#include "../../transform/ir_gen_helpers.h"

#include <assert.h>

#include "spirv/unified1/NonSemanticDebugPrintf.h"
#include "spirv/unified1/GLSL.std.450.h"

#pragma GCC diagnostic error "-Wswitch"

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

    [eq_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpIEqual, SpvOpIEqual, SpvOpFOrdEqual, SpvOpLogicalEqual    }},
    [neq_op] = {Plain, FirstOp, Bool, .fo = {SpvOpINotEqual, SpvOpINotEqual, SpvOpFOrdNotEqual, SpvOpLogicalNotEqual }},
    [lt_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpSLessThan, SpvOpULessThan, SpvOpFOrdLessThan, ISEL_IDENTITY        }},
    [lte_op] = {Plain, FirstOp, Bool, .fo = {SpvOpSLessThanEqual, SpvOpULessThanEqual, SpvOpFOrdLessThanEqual, ISEL_IDENTITY        }},
    [gt_op]  = {Plain, FirstOp, Bool, .fo = {SpvOpSGreaterThan, SpvOpUGreaterThan, SpvOpFOrdGreaterThan, ISEL_IDENTITY        }},
    [gte_op] = {Plain, FirstOp, Bool, .fo = {SpvOpSGreaterThanEqual, SpvOpUGreaterThanEqual, SpvOpFOrdGreaterThanEqual, ISEL_IDENTITY        }},

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
        { SpvOpConvertPtrToU, SpvOpConvertPtrToU, ISEL_ILLEGAL,  ISEL_ILLEGAL,  ISEL_IDENTITY }
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

    [min_op] = {Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMin, (SpvOp) GLSLstd450UMin, (SpvOp) GLSLstd450FMin, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [max_op] = {Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMax, (SpvOp) GLSLstd450UMax, (SpvOp) GLSLstd450FMax, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [exp_op] = {Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Exp},
    [pow_op] = {Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Pow},

    [debug_printf_op] = {Plain, Monomorphic, Void, .extended_set = "NonSemantic.DebugPrintf", .op = (SpvOp) NonSemanticDebugPrintfDebugPrintf},

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

static SpvId emit_primop(Emitter* emitter, FnBuilder fn_builder, BBBuilder bb_builder, const Node* instr) {
    PrimOp the_op = instr->payload.prim_op;
    Nodes args = the_op.operands;
    Nodes type_arguments = the_op.type_arguments;

    IselTableEntry entry = isel_table[the_op.op];
    if (entry.class != Custom) {
        LARRAY(SpvId, emitted_args, args.count);
        for (size_t i = 0; i < args.count; i++)
            emitted_args[i] = emit_value(emitter, bb_builder, args.nodes[i]);

        switch (entry.class) {
            case Plain: {
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                if (opcode == SpvOpNop) {
                    return emitted_args[0];
                }

                Nodes results_ts = unwrap_multiple_yield_types(emitter->arena, instr->type);
                SpvId result_t = results_ts.count >= 1 ? emit_type(emitter, instr->type) : emitter->void_t;

                assert(opcode != SpvOpMax);

                if (entry.extended_set) {
                    SpvId set_id = get_extended_instruction_set(emitter, entry.extended_set);

                    SpvId result = spvb_ext_instruction(bb_builder, result_t, set_id, opcode, args.count, emitted_args);
                    return result;
                } else {
                    SpvId result = spvb_op(bb_builder, opcode, result_t, args.count, emitted_args);
                    return result;
                }
                break;
            }
            case Custom: break;
        }
        SHADY_UNREACHABLE;
    }
    switch (the_op.op) {
        case subgroup_ballot_op: {
            const Type* i32x4 = pack_type(emitter->arena, (PackType) { .width = 4, .element_type = uint32_type(emitter->arena) });
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId raw_result = spvb_group_ballot(bb_builder, emit_type(emitter, i32x4), emit_value(emitter, bb_builder, first(args)), scope_subgroup);
            // TODO: why are we doing this in SPIR-V and not the IR ?
            SpvId low32 = spvb_extract(bb_builder, emit_type(emitter, uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 0 });
            SpvId hi32 = spvb_extract(bb_builder, emit_type(emitter, uint32_type(emitter->arena)), raw_result, 1, (uint32_t[]) { 1 });
            SpvId low64 = spvb_op(bb_builder, SpvOpUConvert, emit_type(emitter, uint64_type(emitter->arena)), 1, &low32);
            SpvId hi64 = spvb_op(bb_builder, SpvOpUConvert, emit_type(emitter, uint64_type(emitter->arena)), 1, &hi32);
            hi64 = spvb_op(bb_builder, SpvOpShiftLeftLogical, emit_type(emitter, uint64_type(emitter->arena)), 2, (SpvId []) { hi64, emit_value(emitter, bb_builder, int64_literal(emitter->arena, 32)) });
            SpvId final_result = spvb_op(bb_builder, SpvOpBitwiseOr, emit_type(emitter, uint64_type(emitter->arena)), 2, (SpvId []) { low64, hi64 });
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
            return final_result;
        }
        case subgroup_broadcast_first_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result;

            if (emitter->configuration->hacks.spv_shuffle_instead_of_broadcast_first) {
                const Node* b = ref_decl_helper(emitter->arena, get_builtin(emitter->module, BuiltinSubgroupLocalInvocationId, NULL));
                SpvId local_id = emit_primop(emitter, fn_builder, bb_builder, prim_op(emitter->arena, (PrimOp) { .op = load_op, .operands = singleton(b) }));
                result = spvb_group_shuffle(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), scope_subgroup, emit_value(emitter, bb_builder, first(args)), local_id);
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformShuffle);
            } else {
                result = spvb_group_broadcast_first(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup);
            }

            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
            return result;
        }
        case subgroup_reduce_sum_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformArithmetic);
            return spvb_group_non_uniform_iadd(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup, SpvGroupOperationReduce, NULL);
        }
        case subgroup_elect_first_op: {
            SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniform);
            return spvb_group_elect(bb_builder, result_t, scope_subgroup);
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

            SpvId result;
            if (dynamic) {
                LARRAY(SpvId, indices, indices_count);
                for (size_t i = 0; i < indices_count; i++) {
                    indices[i] = emit_value(emitter, bb_builder, args.nodes[i + indices_start]);
                }
                assert(indices_count == 1);
                result = spvb_vector_extract_dynamic(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, src_value), indices[0]);
            } else {
                LARRAY(uint32_t, indices, indices_count);
                for (size_t i = 0; i < indices_count; i++) {
                    // TODO: fallback to Dynamic variants transparently
                    indices[i] = get_int_literal_value(*resolve_to_int_literal(args.nodes[i + indices_start]), false);
                }

                if (!insert) {
                    result = spvb_extract(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, src_value), indices_count, indices);
                } else
                    result = spvb_insert(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, args.nodes[1]), emit_value(emitter, bb_builder, src_value), indices_count, indices);
            }
            return result;
        }
        case shuffle_op: {
            const Type* result_t = instr->type;
            SpvId a = emit_value(emitter, bb_builder, args.nodes[0]);
            SpvId b = emit_value(emitter, bb_builder, args.nodes[1]);
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
        case load_op: {
            const Type* ptr_type = first(args)->type;
            deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobalPhysical) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            return spvb_load(bb_builder, emit_type(emitter, elem_type), eptr, operands_count, operands);
        }
        case store_op: {
            const Type* ptr_type = first(args)->type;
            deconstruct_qualified_type(&ptr_type);
            assert(ptr_type->tag == PtrType_TAG);
            const Type* elem_type = ptr_type->payload.ptr_type.pointed_type;

            size_t operands_count = 0;
            uint32_t operands[2];
            if (ptr_type->payload.ptr_type.address_space == AsGlobalPhysical) {
                // TODO only do this in VK mode ?
                TypeMemLayout layout = get_mem_layout(emitter->arena, elem_type);
                operands[operands_count + 0] = SpvMemoryAccessAlignedMask;
                operands[operands_count + 1] = (uint32_t) layout.alignment_in_bytes;
                operands_count += 2;
            }

            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            SpvId eval = emit_value(emitter, bb_builder, args.nodes[1]);
            spvb_store(bb_builder, eval, eptr, operands_count, operands);
            return 0;
        }
        case alloca_logical_op: {
            const Type* elem_type = first(type_arguments);
            return spvb_local_variable(fn_builder, emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunctionLogical,
                .pointed_type = elem_type
            })), SpvStorageClassFunction);
        }
        case lea_op: {
            SpvId base = emit_value(emitter, bb_builder, first(args));

            LARRAY(SpvId, indices, args.count - 2);
            for (size_t i = 2; i < args.count; i++)
                indices[i - 2] = args.nodes[i] ? emit_value(emitter, bb_builder, args.nodes[i]) : 0;

            const IntLiteral* known_offset = resolve_to_int_literal(args.nodes[1]);
            if (known_offset && known_offset->value == 0) {
                const Type* target_type = instr->type;
                return spvb_access_chain(bb_builder, emit_type(emitter, target_type), base, args.count - 2, indices);
            } else {
                error("TODO: OpPtrAccessChain")
            }
            SHADY_UNREACHABLE;
        }
        case select_op: {
            SpvId cond = emit_value(emitter, bb_builder, first(args));
            SpvId truv = emit_value(emitter, bb_builder, args.nodes[1]);
            SpvId flsv = emit_value(emitter, bb_builder, args.nodes[2]);

            return spvb_select(bb_builder, emit_type(emitter, args.nodes[1]->type), cond, truv, flsv);
        }
        default: error("TODO: unhandled op");
    }
    error("unreachable");
}

static SpvId emit_leaf_call(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, BBBuilder bb_builder, Call call) {
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
        args[i] = emit_value(emitter, bb_builder, call.args.nodes[i]);
    SpvId result = spvb_call(bb_builder, return_type, callee, call.args.count, args);
    return result;
}

static SpvId emit_compound_instruction(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, const Node* instruction) {
    Nodes list = instruction->payload.compound_instruction.instructions;
    SpvId id = 0;
    for (size_t i = 0; i < list.count; i++)
        id = emit_instruction(emitter, fn_builder, bb_builder, instruction);
    return id;
}

SpvId emit_instruction(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, const Node* instruction) {
    assert(instruction && is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: error("");
        case CompoundInstruction_TAG: return emit_compound_instruction(emitter, fn_builder, bb_builder, instruction);
        case Call_TAG:                return emit_leaf_call(emitter, fn_builder, *bb_builder, instruction->payload.call);
        case PrimOp_TAG:              return emit_primop(emitter, fn_builder, *bb_builder, instruction);
        case Comment_TAG:             return 0;
    }
    SHADY_UNREACHABLE;
}
