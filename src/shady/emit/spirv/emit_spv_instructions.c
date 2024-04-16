#include "emit_spv.h"

#include "log.h"
#include "portability.h"

#include "../../type.h"
#include "../../transform/memory_layout.h"
#include "../../transform/ir_gen_helpers.h"

#include <assert.h>

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

    [min_op] = { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMin, (SpvOp) GLSLstd450UMin, (SpvOp) GLSLstd450FMin, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [max_op] = { Plain, FirstOp, Same, .extended_set = "GLSL.std.450", .fo = {(SpvOp) GLSLstd450SMax, (SpvOp) GLSLstd450UMax, (SpvOp) GLSLstd450FMax, ISEL_ILLEGAL, ISEL_ILLEGAL }},
    [exp_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Exp },
    [pow_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Pow },
    [fma_op] = { Plain, Monomorphic, Same, .extended_set = "GLSL.std.450", .op = (SpvOp) GLSLstd450Fma },

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

static void emit_primop(Emitter* emitter, FnBuilder fn_builder, BBBuilder bb_builder, const Node* instr, size_t results_count, SpvId results[]) {
    PrimOp the_op = instr->payload.prim_op;
    Nodes args = the_op.operands;
    Nodes type_arguments = the_op.type_arguments;

    IselTableEntry entry = isel_table[the_op.op];
    if (entry.class != Custom) {
        assert(results_count <= 1);
        LARRAY(SpvId, emitted_args, args.count);
        for (size_t i = 0; i < args.count; i++)
            emitted_args[i] = emit_value(emitter, bb_builder, args.nodes[i]);

        switch (entry.class) {
            case Plain: {
                SpvOp opcode = get_opcode(emitter, entry, args, type_arguments);
                if (opcode == SpvOpNop) {
                    assert(results_count == 1);
                    results[0] = emitted_args[0];
                    return;
                }

                Nodes results_ts = unwrap_multiple_yield_types(emitter->arena, instr->type);
                SpvId result_t = results_ts.count >= 1 ? emit_type(emitter, instr->type) : emitter->void_t;

                assert(opcode != SpvOpMax);

                if (entry.extended_set) {
                    SpvId set_id = get_extended_instruction_set(emitter, entry.extended_set);

                    SpvId result = spvb_ext_instruction(bb_builder, result_t, set_id, opcode, args.count, emitted_args);
                    if (results_count == 1)
                        results[0] = result;
                } else {
                    SpvId result = spvb_op(bb_builder, opcode, result_t, args.count, emitted_args);
                    if (results_count == 1)
                        results[0] = result;
                }
                return;
            }
            case Custom: SHADY_UNREACHABLE;
        }

        return;
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
            assert(results_count == 1);
            results[0] = final_result;
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
            return;
        }
        case subgroup_broadcast_first_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result;

            if (emitter->configuration->hacks.spv_shuffle_instead_of_broadcast_first) {
                SpvId local_id;
                const Node* b = ref_decl_helper(emitter->arena, get_or_create_builtin(emitter->module, BuiltinSubgroupLocalInvocationId, NULL));
                emit_primop(emitter, fn_builder, bb_builder, prim_op(emitter->arena, (PrimOp) { .op = load_op, .operands = singleton(b) }), 1, &local_id);
                result = spvb_group_shuffle(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), scope_subgroup, emit_value(emitter, bb_builder, first(args)), local_id);
                spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformShuffle);
            } else {
                result = spvb_group_broadcast_first(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup);
            }

            assert(results_count == 1);
            results[0] = result;
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformBallot);
            return;
        }
        case subgroup_reduce_sum_op: {
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            assert(results_count == 1);
            results[0] = spvb_group_non_uniform_iadd(bb_builder, emit_type(emitter, get_unqualified_type(first(args)->type)), emit_value(emitter, bb_builder, first(args)), scope_subgroup, SpvGroupOperationReduce, NULL);
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniformArithmetic);
            return;
        }
        case subgroup_elect_first_op: {
            SpvId result_t = emit_type(emitter, bool_type(emitter->arena));
            SpvId scope_subgroup = emit_value(emitter, bb_builder, int32_literal(emitter->arena, SpvScopeSubgroup));
            SpvId result = spvb_group_elect(bb_builder, result_t, scope_subgroup);
            assert(results_count == 1);
            results[0] = result;
            spvb_capability(emitter->file_builder, SpvCapabilityGroupNonUniform);
            return;
        }
        case insert_op:
        case extract_dynamic_op:
        case extract_op: {
            assert(results_count == 1);
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
                    indices[i] = emit_value(emitter, bb_builder, args.nodes[i + indices_start]);
                }
                assert(indices_count == 1);
                results[0] = spvb_vector_extract_dynamic(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, src_value), indices[0]);
            } else {
                LARRAY(uint32_t, indices, indices_count);
                for (size_t i = 0; i < indices_count; i++) {
                    // TODO: fallback to Dynamic variants transparently
                    indices[i] = get_int_literal_value(*resolve_to_int_literal(args.nodes[i + indices_start]), false);
                }

                if (!insert) {
                    results[0] = spvb_extract(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, src_value), indices_count, indices);
                } else
                    results[0] = spvb_insert(bb_builder, emit_type(emitter, result_t), emit_value(emitter, bb_builder, args.nodes[1]), emit_value(emitter, bb_builder, src_value), indices_count, indices);
            }
            return;
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
            assert(results_count == 1);
            results[0] = spvb_vecshuffle(bb_builder, emit_type(emitter, result_t), a, b, args.count - 2, indices);
            return;
        }
        case load_op: {
            const Type* ptr_type = first(args)->type;
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

            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            SpvId result = spvb_load(bb_builder, emit_type(emitter, elem_type), eptr, operands_count, operands);
            assert(results_count == 1);
            results[0] = result;
            return;
        }
        case store_op: {
            const Type* ptr_type = first(args)->type;
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

            SpvId eptr = emit_value(emitter, bb_builder, first(args));
            SpvId eval = emit_value(emitter, bb_builder, args.nodes[1]);
            spvb_store(bb_builder, eval, eptr, operands_count, operands);
            assert(results_count == 0);
            return;
        }
        case alloca_logical_op: {
            const Type* elem_type = first(type_arguments);
            SpvId result = spvb_local_variable(fn_builder, emit_type(emitter, ptr_type(emitter->arena, (PtrType) {
                .address_space = AsFunction,
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

            const IntLiteral* known_offset = resolve_to_int_literal(args.nodes[1]);
            if (known_offset && known_offset->value == 0) {
                const Type* target_type = instr->type;
                SpvId result = spvb_access_chain(bb_builder, emit_type(emitter, target_type), base, args.count - 2, indices);
                assert(results_count == 1);
                results[0] = result;
            } else {
                const Type* target_type = instr->type;
                SpvId result = spvb_ptr_access_chain(bb_builder, emit_type(emitter, target_type), base, emit_value(emitter, bb_builder, args.nodes[1]), args.count - 2, indices);
                assert(results_count == 1);
                results[0] = result;
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
        default: error("TODO: unhandled op");
    }
    error("unreachable");
}

static void emit_leaf_call(Emitter* emitter, SHADY_UNUSED FnBuilder fn_builder, BBBuilder bb_builder, Call call, size_t results_count, SpvId results[]) {
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
    LARRAY(SpvbPhi*, join_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        assert(if_instr.if_false && "Ifs with yield types need false branches !");
        SpvId phi_id = spvb_fresh_id(emitter->file_builder);
        SpvId type = emit_type(emitter, yield_types.nodes[i]);
        SpvbPhi* phi = spvb_add_phi(join_bb, type, phi_id);
        join_phis[i] = phi;
        results[i] = phi_id;
    }

    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.join_target = join_bb_id;
    merge_targets_branches.join_phis = join_phis;

    BBBuilder true_bb = spvb_begin_bb(fn_builder, true_id);
    spvb_add_bb(fn_builder, true_bb);
    assert(is_case(if_instr.if_true));
    emit_terminator(emitter, fn_builder, true_bb, merge_targets_branches, if_instr.if_true->payload.case_.body);
    if (if_instr.if_false) {
        BBBuilder false_bb = spvb_begin_bb(fn_builder, false_id);
        spvb_add_bb(fn_builder, false_bb);
        assert(is_case(if_instr.if_false));
        emit_terminator(emitter, fn_builder, false_bb, merge_targets_branches, if_instr.if_false->payload.case_.body);
    }

    spvb_add_bb(fn_builder, join_bb);
    *bb_builder = join_bb;
}

static void emit_match(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Match match, size_t results_count, SHADY_UNUSED SpvId results[]) {
    Nodes yield_types = match.yield_types;
    assert(yield_types.count == results_count);
    SpvId join_bb_id = spvb_fresh_id(emitter->file_builder);

    assert(get_unqualified_type(match.inspect->type)->tag == Int_TAG);
    SpvId inspectee = emit_value(emitter, *bb_builder, match.inspect);

    SpvId default_id = spvb_fresh_id(emitter->file_builder);

    const Type* inspectee_t = match.inspect->type;
    deconstruct_qualified_type(&inspectee_t);
    assert(inspectee_t->tag == Int_TAG);
    size_t literal_width = inspectee_t->payload.int_type.width == IntTy64 ? 2 : 1;
    size_t literal_case_entry_size = literal_width + 1;
    LARRAY(uint32_t, literals_and_cases, match.cases.count * literal_case_entry_size);
    for (size_t i = 0; i < match.cases.count; i++) {
        uint64_t value = (uint64_t) get_int_literal_value(*resolve_to_int_literal(match.literals.nodes[i]), false);
        if (inspectee_t->payload.int_type.width == IntTy64) {
            literals_and_cases[i * literal_case_entry_size + 0] = (SpvId) (uint32_t) (value & 0xFFFFFFFF);
            literals_and_cases[i * literal_case_entry_size + 1] = (SpvId) (uint32_t) (value >> 32);
        } else {
            literals_and_cases[i * literal_case_entry_size + 0] = (SpvId) (uint32_t) value;
        }
        literals_and_cases[i * literal_case_entry_size + literal_width] = spvb_fresh_id(emitter->file_builder);
    }

    spvb_selection_merge(*bb_builder, join_bb_id, 0);
    spvb_switch(*bb_builder, inspectee, default_id, match.cases.count * literal_case_entry_size, literals_and_cases);

    // When 'join' is codegen'd, these will be filled with the values given to it
    BBBuilder join_bb = spvb_begin_bb(fn_builder, join_bb_id);
    LARRAY(SpvbPhi*, join_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        SpvId phi_id = spvb_fresh_id(emitter->file_builder);
        SpvId type = emit_type(emitter, yield_types.nodes[i]);
        SpvbPhi* phi = spvb_add_phi(join_bb, type, phi_id);
        join_phis[i] = phi;
        results[i] = phi_id;
    }

    MergeTargets merge_targets_branches = *merge_targets;
    merge_targets_branches.join_target = join_bb_id;
    merge_targets_branches.join_phis = join_phis;

    for (size_t i = 0; i < match.cases.count; i++) {
        BBBuilder case_bb = spvb_begin_bb(fn_builder, literals_and_cases[i * literal_case_entry_size + literal_width]);
        const Node* case_body = match.cases.nodes[i];
        assert(is_case(case_body));
        spvb_add_bb(fn_builder, case_bb);
        emit_terminator(emitter, fn_builder, case_bb, merge_targets_branches, case_body->payload.case_.body);
    }
    BBBuilder default_bb = spvb_begin_bb(fn_builder, default_id);
    assert(is_case(match.default_case));
    spvb_add_bb(fn_builder, default_bb);
    emit_terminator(emitter, fn_builder, default_bb, merge_targets_branches, match.default_case->payload.case_.body);

    spvb_add_bb(fn_builder, join_bb);
    *bb_builder = join_bb;
}

static void emit_loop(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, Loop loop_instr, size_t results_count, SpvId results[]) {
    Nodes yield_types = loop_instr.yield_types;
    assert(yield_types.count == results_count);

    const Node* body = loop_instr.body;
    assert(is_case(body));
    Nodes body_params = body->payload.case_.params;

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
    LARRAY(SpvbPhi*, loop_break_phis, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++) {
        SpvId yielded_type = emit_type(emitter, get_unqualified_type(yield_types.nodes[i]));

        SpvId break_phi_id = spvb_fresh_id(emitter->file_builder);
        SpvbPhi* phi = spvb_add_phi(next, yielded_type, break_phi_id);
        loop_break_phis[i] = phi;
        results[i] = break_phi_id;
    }

    // Wire up the phi nodes for the loop contents
    LARRAY(SpvbPhi*, loop_continue_phis, body_params.count);
    for (size_t i = 0; i < body_params.count; i++) {
        SpvId loop_param_type = emit_type(emitter, get_unqualified_type(body_params.nodes[i]->type));

        SpvId continue_phi_id = spvb_fresh_id(emitter->file_builder);
        SpvbPhi* continue_phi = spvb_add_phi(continue_builder, loop_param_type, continue_phi_id);
        loop_continue_phis[i] = continue_phi;

        // To get the actual loop parameter, we make a second phi for the nodes that go into the header
        // We already know the two edges into the header so we immediately add the Phi sources for it.
        SpvId loop_param_id = spvb_fresh_id(emitter->file_builder);
        SpvbPhi* loop_param_phi = spvb_add_phi(header_builder, loop_param_type, loop_param_id);
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
    emit_terminator(emitter, fn_builder, body_builder, merge_targets_branches, body->payload.case_.body);

    // the continue block just jumps back into the header
    spvb_branch(continue_builder, header_id);
    spvb_add_bb(fn_builder, continue_builder);

    // We start the next block
    spvb_add_bb(fn_builder, next);
    *bb_builder = next;
}

void emit_instruction(Emitter* emitter, FnBuilder fn_builder, BBBuilder* bb_builder, MergeTargets* merge_targets, const Node* instruction, size_t results_count, SpvId results[]) {
    assert(is_instruction(instruction));

    switch (is_instruction(instruction)) {
        case NotAnInstruction: error("");
        case Instruction_Control_TAG:
        case Instruction_Block_TAG: error("Should be lowered elsewhere")
        case Instruction_Call_TAG: emit_leaf_call(emitter, fn_builder, *bb_builder, instruction->payload.call, results_count, results);                 break;
        case PrimOp_TAG:              emit_primop(emitter, fn_builder, *bb_builder, instruction, results_count, results);                                    break;
        case If_TAG:                      emit_if(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.if_instr, results_count, results);    break;
        case Match_TAG:                emit_match(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.match_instr, results_count, results); break;
        case Loop_TAG:                  emit_loop(emitter, fn_builder, bb_builder, merge_targets, instruction->payload.loop_instr, results_count, results);  break;
        case Comment_TAG: break;
        default: error("TODO: unhandled instruction");
    }
}
