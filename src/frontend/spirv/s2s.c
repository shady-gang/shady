#include "s2s.h"
#include "shady/builtins.h"

// this avoids polluting the namespace
#define SpvHasResultAndType ShadySpvHasResultAndType

// hacky nonsense because this guy is not declared static
typedef enum SpvOp_ SpvOp;
static void SpvHasResultAndType(SpvOp opcode, bool *hasResult, bool *hasResultType);

#define SPV_ENABLE_UTILITY_CODE 1
#include "spirv/unified1/spirv.h"
#include "spirv/unified1/OpenCL.std.h"
#include "spirv/unified1/GLSL.std.450.h"

extern SpvBuiltIn spv_builtins[];

// TODO: reserve real decoration IDs
typedef enum {
    ShdDecorationName           = 999999,
    ShdDecorationMemberName     = 999998,
    ShdDecorationEntryPointType = 999997,
    ShdDecorationEntryPointName = 999996,
} ShdDecoration;

#include "log.h"
#include "arena.h"
#include "portability.h"
#include "dict.h"
#include "util.h"

#include "../shady/type.h"
#include "../shady/ir_private.h"
#include "../shady/transform/ir_gen_helpers.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    struct {
        uint8_t major, minor;
    } version;
    uint32_t generator;
    uint32_t bound;
} SpvHeader;

typedef struct SpvDeco_ SpvDeco;

typedef struct {
    enum { Nothing, Forward, Str, Typ, Decl, BB, Value, Literals } type;
    const Type* result_type;
    union {
        size_t instruction_offset;
        const Node* node;
        String str;
        struct { size_t count; uint32_t* data; } literals;
    };
    SpvDeco* next_decoration;
    size_t final_size;
} SpvDef;

struct SpvDeco_ {
    SpvDecoration decoration;
    int member;
    SpvDef payload;
};

typedef struct SpvPhiArgs_ SpvPhiArgs;
struct SpvPhiArgs_ {
    SpvId predecessor;
    size_t arg_i;
    SpvId arg;
    SpvPhiArgs* next_;
};

typedef struct {
    size_t cursor;
    size_t len;
    uint32_t* words;
    Module* mod;
    IrArena* arena;

    bool is_entry_pt;
    Node* fun;
    size_t fun_arg_i;

    struct CurrBlock {
        SpvId id;
        BodyBuilder* builder;
        const Node* finished;
    } current_block;

    SpvHeader header;
    SpvDef* defs;
    Arena* decorations_arena;
    struct Dict* phi_arguments;
} SpvParser;

SpvDef* get_definition_by_id(SpvParser* parser, size_t id);

SpvDef* new_def(SpvParser* parser) {
    SpvDef* interned = arena_alloc(parser->decorations_arena, sizeof(SpvDef));
    SpvDef empty = {0};
    memcpy(interned, &empty, sizeof(SpvDef));
    return interned;
}

void add_decoration(SpvParser* parser, SpvId id, SpvDeco decoration) {
    SpvDef* tgt_def = &parser->defs[id];
    while (tgt_def->next_decoration) {
        tgt_def = &tgt_def->next_decoration->payload;
    }
    SpvDeco* interned = arena_alloc(parser->decorations_arena, sizeof(SpvDeco));
    memcpy(interned, &decoration, sizeof(SpvDeco));
    tgt_def->next_decoration = interned;
}

SpvDeco* find_decoration(SpvParser* parser, SpvId id, int member, SpvDecoration tag) {
    SpvDef* tgt_def = &parser->defs[id];
    while (tgt_def->next_decoration) {
        if (tgt_def->next_decoration->decoration == tag && (member < 0 || tgt_def->next_decoration->member == member))
            return tgt_def->next_decoration;
        tgt_def = &tgt_def->next_decoration->payload;
    }
    return NULL;
}

String get_name(SpvParser* parser, SpvId id) {
    SpvDeco* deco = find_decoration(parser, id, -1, ShdDecorationName);
    if (!deco)
        return NULL;
    return deco->payload.str;
}

String get_member_name(SpvParser* parser, SpvId id, int member_id) {
    SpvDeco* deco = find_decoration(parser, id, member_id, ShdDecorationName);
    if (!deco)
        return NULL;
    return deco->payload.str;
}

const Type* get_def_type(SpvParser* parser, SpvId id) {
    SpvDef* def = get_definition_by_id(parser, id);
    assert(def->type == Typ);
    const Node* t = def->node;
    assert(t && is_type(t));
    return t;
}

const Type* get_def_decl(SpvParser* parser, SpvId id) {
    SpvDef* def = get_definition_by_id(parser, id);
    assert(def->type == Decl);
    const Node* n = def->node;
    assert(n && is_declaration(n));
    return n;
}

String get_def_string(SpvParser* parser, SpvId id) {
    SpvDef* def = get_definition_by_id(parser, id);
    assert(def->type == Str);
    return def->str;
}

const Type* get_def_ssa_value(SpvParser* parser, SpvId id) {
    SpvDef* def = get_definition_by_id(parser, id);
    const Node* n = def->node;
    if (is_declaration(n))
        n = ref_decl_helper(parser->arena, n);
    assert(n && (is_value(n) || is_instruction(n)));
    return n;
}

const Type* get_def_block(SpvParser* parser, SpvId id) {
    SpvDef* def = get_definition_by_id(parser, id);
    assert(def->type == BB);
    const Node* n = def->node;
    assert(n && n->tag == BasicBlock_TAG);
    return n;
}

bool parse_spv_header(SpvParser* parser) {
    assert(parser->cursor == 0);
    assert(parser->len >= 4);
    assert(parser->words[0] == SpvMagicNumber);

    parser->header.version.major = (uint8_t) (parser->words[1] >> 16);
    parser->header.version.minor = (uint8_t) (parser->words[1] >>  8);

    parser->header.generator = parser->words[2];
    parser->header.bound = parser->words[3];
    assert(parser->words[4] == 0);
    parser->cursor = 5;
    return true;
}

String decode_spv_string_literal(SpvParser* parser, uint32_t* at) {
    // TODO: assumes little endian
    return string(get_module_arena(parser->mod), (const char*) at);
}

AddressSpace convert_storage_class(SpvStorageClass class) {
    switch (class) {
        case SpvStorageClassInput:                 return AsInput;
        case SpvStorageClassOutput:                return AsOutput;
        case SpvStorageClassWorkgroup:             return AsShared;
        case SpvStorageClassCrossWorkgroup:        return AsGlobal;
        case SpvStorageClassPhysicalStorageBuffer: return AsGlobal;
        case SpvStorageClassPrivate:               return AsPrivate;
        case SpvStorageClassFunction:              return AsPrivate;
        case SpvStorageClassGeneric:               return AsGeneric;
        case SpvStorageClassPushConstant:          return AsPushConstant;
        case SpvStorageClassAtomicCounter:
            break;
        case SpvStorageClassImage:                 return AsImage;
            break;
        case SpvStorageClassStorageBuffer:         return AsShaderStorageBufferObject;
        case SpvStorageClassUniformConstant:
        case SpvStorageClassUniform:               return AsGlobal; // TODO: should probably depend on CL/VK flavours!
        case SpvStorageClassCallableDataKHR:
        case SpvStorageClassIncomingCallableDataKHR:
        case SpvStorageClassRayPayloadKHR:
        case SpvStorageClassHitAttributeKHR:
        case SpvStorageClassIncomingRayPayloadKHR:
        case SpvStorageClassShaderRecordBufferKHR:
            break;
        case SpvStorageClassCodeSectionINTEL:
        case SpvStorageClassDeviceOnlyINTEL:
        case SpvStorageClassHostOnlyINTEL:
        case SpvStorageClassMax:
            break;
        default:
            break;
    }
    error("s2s: Unsupported storage class: %d\n", class);
}

typedef struct {
    bool valid;
    Op op;
    int ops_offset;
} SpvShdOpMapping;

static SpvShdOpMapping spv_shd_op_mapping[] = {
    // 3.42.13 Arithmetic operations
    [SpvOpSNegate] = { 1, neg_op, 3 },
    [SpvOpFNegate] = { 1, neg_op, 3 },
    [SpvOpIAdd] = { 1, add_op, 3 },
    [SpvOpFAdd] = { 1, add_op, 3 },
    [SpvOpISub] = { 1, sub_op, 3 },
    [SpvOpFSub] = { 1, sub_op, 3 },
    [SpvOpIMul] = { 1, mul_op, 3 },
    [SpvOpFMul] = { 1, mul_op, 3 },
    [SpvOpUDiv] = { 1, div_op, 3 },
    [SpvOpSDiv] = { 1, div_op, 3 },
    [SpvOpFDiv] = { 1, div_op, 3 },
    [SpvOpUMod] = { 1, mod_op, 3 },
    [SpvOpSRem] = { 1, mod_op, 3 },
    [SpvOpSMod] = { 1, mod_op, 3 }, /* TODO: this is slightly incorrect! rem and mod are different ops for signed numbers. */
    [SpvOpFRem] = { 1, mod_op, 3 },
    [SpvOpFMod] = { 1, mod_op, 3 }, /* TODO ditto */
    [SpvOpVectorTimesScalar] = { 0 },
    [SpvOpMatrixTimesScalar] = { 0 },
    [SpvOpVectorTimesMatrix] = { 0 },
    [SpvOpMatrixTimesVector] = { 0 },
    [SpvOpMatrixTimesMatrix] = { 0 },
    [SpvOpOuterProduct] = { 0 },
    [SpvOpDot] = { 0 },
    [SpvOpIAddCarry] = { 1, add_carry_op, 3},
    [SpvOpISubBorrow] = { 1, sub_borrow_op, 3},
    [SpvOpUMulExtended] = { 1, mul_extended_op, 3},
    [SpvOpSMulExtended] = { 1, mul_extended_op, 3},
    [SpvOpSDot] = { 0 },
    [SpvOpUDot] = { 0 },
    [SpvOpSUDot] = { 0 },
    [SpvOpSDotAccSat] = { 0 },
    [SpvOpUDotAccSat] = { 0 },
    [SpvOpSUDotAccSat] = { 0 },
    // 3.42.14 Bit instructions
    [SpvOpShiftRightLogical] = { 1, rshift_logical_op, 3 },
    [SpvOpShiftRightArithmetic] = { 1, rshift_arithm_op, 3 },
    [SpvOpShiftLeftLogical] = { 1, lshift_op, 3 },
    [SpvOpBitwiseOr] = { 1, or_op, 3 },
    [SpvOpBitwiseXor] = { 1, xor_op, 3 },
    [SpvOpBitwiseAnd] = { 1, and_op, 3 },
    [SpvOpNot] = { 1, not_op, 3 },
    [SpvOpBitFieldInsert] = { 0 },
    [SpvOpBitFieldSExtract] = { 0 },
    [SpvOpBitFieldUExtract] = { 0 },
    [SpvOpBitReverse] = { 0 },
    [SpvOpBitCount] = { 0 },
    // 3.42.15 Relational and Logical instructions
    [SpvOpAny] = { 0 },
    [SpvOpAll] = { 0 },
    [SpvOpIsNan] = { 0 },
    [SpvOpIsInf] = { 0 },
    [SpvOpIsFinite] = { 0 },
    [SpvOpIsNormal] = { 0 },
    [SpvOpSignBitSet] = { 0 },
    [SpvOpLessOrGreater] = { 0 },
    [SpvOpOrdered] = { 0 },
    [SpvOpUnordered] = { 0 },
    [SpvOpLogicalEqual] = { 1, eq_op, 3 },
    [SpvOpLogicalNotEqual] = { 1, neq_op, 3 },
    [SpvOpLogicalOr] = { 1, or_op, 3 },
    [SpvOpLogicalAnd] = { 1, and_op, 3 },
    [SpvOpLogicalNot] = { 1, not_op, 3 },
    [SpvOpSelect] = { 1, select_op, 3 },
    [SpvOpIEqual] = { 1, eq_op, 3 },
    [SpvOpINotEqual] = { 1, neq_op, 3 },
    [SpvOpUGreaterThan] = { 1, gt_op, 3 },
    [SpvOpSGreaterThan] = { 1, gt_op, 3 },
    [SpvOpUGreaterThanEqual] = { 1, gte_op, 3 },
    [SpvOpSGreaterThanEqual] = { 1, gte_op, 3 },
    [SpvOpULessThan] = { 1, lt_op, 3 },
    [SpvOpSLessThan] = { 1, lt_op, 3 },
    [SpvOpULessThanEqual] = { 1, lte_op, 3 },
    [SpvOpSLessThanEqual] = { 1, lte_op, 3 },
    [SpvOpFOrdEqual] = { 1, eq_op, 3 },
    [SpvOpFUnordEqual] = { 1, eq_op, 3 }, /* TODO again these are not the same */
    [SpvOpFOrdNotEqual] = { 1, neq_op, 3 },
    [SpvOpFUnordNotEqual] = { 1, neq_op, 3 }, /* ditto */
    [SpvOpFOrdLessThan] = { 1, lt_op, 3 },
    [SpvOpFUnordLessThan] = { 1, lt_op, 3 },
    [SpvOpFOrdLessThanEqual] = { 1, lte_op, 3 },
    [SpvOpFUnordLessThanEqual] = { 1, lte_op, 3 },
    [SpvOpFOrdGreaterThan] = { 1, gt_op, 3 },
    [SpvOpFUnordGreaterThan] = { 1, gt_op, 3 },
    [SpvOpFOrdGreaterThanEqual] = { 1, gte_op, 3 },
    [SpvOpFUnordGreaterThanEqual] = { 1, gte_op, 3 },
    // 3.42.16 Derivative Instructions
    // honestly none of those are implemented ...
};

const SpvShdOpMapping* convert_spv_op(SpvOp src) {
    const int nentries = sizeof(spv_shd_op_mapping) / sizeof(*spv_shd_op_mapping);
    if (src >= nentries)
        return NULL;
    if (spv_shd_op_mapping[src].valid)
        return &spv_shd_op_mapping[src];
    return NULL;
}

SpvId get_result_defined_at(SpvParser* parser, size_t instruction_offset) {
    uint32_t* instruction = parser->words + instruction_offset;

    SpvOp op = instruction[0] & 0xFFFF;
    SpvId result;
    bool has_result, has_type;
    SpvHasResultAndType(op, &has_result, &has_type);
    if (has_result) {
        if (has_type)
            result = instruction[2];
        else
            result = instruction[1];
        return result;
    }
    error("no result defined at offset %zu", instruction_offset);
}

void scan_definitions(SpvParser* parser) {
    size_t old_cursor = parser->cursor;
    while (true) {
        size_t available = parser->len - parser->cursor;
        if (available == 0)
            break;
        assert(available > 0);
        uint32_t* instruction = parser->words + parser->cursor;
        SpvOp op = instruction[0] & 0xFFFF;
        int size = (int) ((instruction[0] >> 16u) & 0xFFFFu);

        SpvId result;
        bool has_result, has_type;
        SpvHasResultAndType(op, &has_result, &has_type);
        if (has_result) {
            if (has_type)
                result = instruction[2];
            else
                result = instruction[1];

            parser->defs[result].type = Forward;
            parser->defs[result].instruction_offset = parser->cursor;
        }
        parser->cursor += size;
    }
    parser->cursor = old_cursor;
}

Nodes get_args_from_phi(SpvParser* parser, SpvId block, SpvId predecessor) {
    SpvDef* block_def = get_definition_by_id(parser, block);
    assert(block_def->type == BB && block_def->node);
    int params_count = block_def->node->payload.basic_block.params.count;

    LARRAY(const Node*, params, params_count);
    for (size_t i = 0; i < params_count; i++)
        params[i] = NULL;

    if (params_count == 0)
        return empty(parser->arena);

    SpvPhiArgs** found = find_value_dict(SpvId, SpvPhiArgs*, parser->phi_arguments, block);
    assert(found);
    SpvPhiArgs* arg = *found;
    while (true) {
        if (arg->predecessor == predecessor) {
            assert(arg->arg_i < params_count);
            params[arg->arg_i] = get_def_ssa_value(parser, arg->arg);
        }
        if (arg->next_)
           arg = arg->next_;
        else
            break;
    }

    for (size_t i = 0; i < params_count; i++)
        assert(params[i]);

    return nodes(parser->arena, params_count, params);
}

size_t parse_spv_instruction_at(SpvParser* parser, size_t instruction_offset) {
    IrArena* a = parser->arena;
    uint32_t* instruction = parser->words + instruction_offset;
    SpvOp op = instruction[0] & 0xFFFF;
    int size = (int) ((instruction[0] >> 16u) & 0xFFFFu);
    assert(size > 0);

    SpvId result_t, result;
    bool has_result, has_type;
    SpvHasResultAndType(op, &has_result, &has_type);
    if (has_type) {
        result_t = instruction[1];
        if (has_result)
            result = instruction[2];
    } else if (has_result)
        result = instruction[1];

    if (has_result) {
        assert(parser->defs[result].type != Nothing && "consistency issue");
        if (parser->defs[result].final_size > 0) {
            // already parsed, skip it
            return parser->defs[result].final_size;
        } else if (parser->defs[result].type != Forward) {
            return SIZE_MAX;
        }

        if (has_type)
            parser->defs[result].result_type = get_def_type(parser, result_t);
    }

    instruction_offset += size;

    if (convert_spv_op(op)) {
        assert(parser->current_block.builder);
        SpvShdOpMapping shd_op = *convert_spv_op(op);
        int num_ops = size - shd_op.ops_offset;
        LARRAY(const Node*, ops, num_ops);
        for (size_t i = 0; i < num_ops; i++)
            ops[i] = get_def_ssa_value(parser, instruction[shd_op.ops_offset + i]);
        int results_count = has_result ? 1 : 0;
        Nodes results = bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
            .op = shd_op.op,
            .type_arguments = empty(parser->arena),
            .operands = nodes(parser->arena, num_ops, ops)
        }), results_count);
        if (has_result) {
            parser->defs[result].type = Value;
            parser->defs[result].node = first(results);
        }
        return size;
    }

    switch (op) {
        // shady doesn't care, the emitter will set those up
        case SpvOpMemoryModel:
        case SpvOpCapability: break;
        // these are basically just strings and we can infer how to handle them from ctx at the uses
        case SpvOpExtInstImport:
        case SpvOpString: {
            parser->defs[result].type = Str;
            parser->defs[result].str = decode_spv_string_literal(parser, instruction + 2);
            break;
        }
        case SpvOpExtension: {
            // TODO: do we care to do anything with enabled exts ?
            String ext = decode_spv_string_literal(parser, instruction + 1);
            break;
        }
        case SpvOpName:
        case SpvOpMemberName: {
            SpvId target = instruction[1];
            ShdDecoration decoration = op == SpvOpName ? ShdDecorationName : ShdDecorationMemberName;
            int name_offset = op == SpvOpName ? 2 : 3;
            SpvDeco deco = {
                .payload = { Str, .str = decode_spv_string_literal(parser, instruction + name_offset), .next_decoration = NULL },
                .decoration = decoration,
                .member = op == SpvOpName ? -1 : (int)instruction[3],
            };
            add_decoration(parser, target, deco);
            break;
        }
        // more debug nonsense
        case SpvOpModuleProcessed:
        case SpvOpSource:
        case SpvOpSourceExtension:
            break;
        case SpvOpEntryPoint: {
            String type;
            switch ((SpvExecutionModel) instruction[1]) {
                case SpvExecutionModelGLCompute:
                case SpvExecutionModelKernel:
                    type = "Compute";
                    break;
                case SpvExecutionModelFragment:
                    type = "Fragment";
                    break;
                case SpvExecutionModelVertex:
                    type = "Vertex";
                    break;
                default:
                    error("Unsupported execution model %d", instruction[1])
            }
            add_decoration(parser, instruction[2], (SpvDeco) {
                .decoration = ShdDecorationEntryPointType,
                .member = -1,
                .payload = {
                    .type = Str,
                    .str = type,
                },
            });
            add_decoration(parser, instruction[2], (SpvDeco) {
                .decoration = ShdDecorationEntryPointName,
                .member = -1,
                .payload = {
                    .type = Str,
                    .str = (const char*) &instruction[3],
                },
            });
            break;
        }
        case SpvOpExecutionMode:
        case SpvOpDecorate:
        case SpvOpMemberDecorate: {
            SpvDef payload = { Literals };
            SpvId target = instruction[1];
            int data_offset = op == SpvOpMemberDecorate ? 4 : 3;
            payload.literals.count = size - data_offset;
            payload.literals.data = instruction + data_offset;
            int member = -1;
            if (op == SpvOpExecutionMode)
                member = -2;
            else if (op == SpvOpMemberDecorate)
                member = instruction[3];
            SpvDeco deco = {
                .payload = payload,
                .member = member,
                .decoration = instruction[data_offset - 1]
            };
            add_decoration(parser, target, deco);
            break;
        }
        case SpvOpTypeVoid: {
            parser->defs[result].type = Typ;
            parser->defs[result].node = unit_type(parser->arena);
            break;
        }
        case SpvOpTypeInt: {
            uint32_t width = instruction[2];
            bool is_signed = instruction[3];
            IntSizes w;
            switch (width) {
                case  8: w = IntTy8;  break;
                case 16: w = IntTy16; break;
                case 32: w = IntTy32; break;
                case 64: w = IntTy64; break;
                default: error("unhandled int width");
            }
            parser->defs[result].type = Typ;
            parser->defs[result].node = int_type(parser->arena, (Int) {
                .width = w,
                .is_signed = is_signed,
            });
            break;
        }
        case SpvOpTypeFloat: {
            uint32_t width = instruction[2];
            FloatSizes w;
            switch (width) {
                case 16: w = FloatTy16; break;
                case 32: w = FloatTy32; break;
                case 64: w = FloatTy64; break;
                default: error("unhandled float width");
            }
            parser->defs[result].type = Typ;
            parser->defs[result].node = float_type(parser->arena, (Float) {
                .width = w,
            });
            break;
        }
        case SpvOpTypeBool: {
            parser->defs[result].type = Typ;
            parser->defs[result].node = bool_type(parser->arena);
            break;
        }
        case SpvOpTypePointer: {
            AddressSpace as = convert_storage_class(instruction[2]);
            const Type* element_t = get_def_type(parser, instruction[3]);
            parser->defs[result].type = Typ;
            parser->defs[result].node = ptr_type(parser->arena, (PtrType) {
                .pointed_type = element_t,
                .address_space = as
            });
            break;
        }
        case SpvOpTypeFunction: {
            parser->defs[result].type = Typ;
            const Type* return_t = get_def_type(parser, instruction[2]);
            LARRAY(const Type*, param_ts, size - 3);
            for (size_t i = 0; i < size - 3; i++)
                param_ts[i] = get_def_type(parser, instruction[3 + i]);
            parser->defs[result].node = fn_type(parser->arena, (FnType) {
                .return_types = (return_t == unit_type(parser->arena)) ? empty(parser->arena) : singleton(return_t),
                .param_types = nodes(parser->arena, size - 3, param_ts)
            });
            break;
        }
        case SpvOpTypeStruct: {
            parser->defs[result].type = Typ;
            String name = get_name(parser, result);
            name = name ? name : unique_name(parser->arena, "struct_type");
            Node* nominal_type_decl = nominal_type(parser->mod, empty(parser->arena), name);
            const Node* nom_t_ref = type_decl_ref(parser->arena, (TypeDeclRef) {
                .decl = nominal_type_decl
            });
            parser->defs[result].node = nom_t_ref;
            int members_count = size - 2;
            LARRAY(String, member_names, members_count);
            LARRAY(const Type*, member_tys, members_count);
            for (size_t i = 0; i < members_count; i++) {
                member_names[i] = get_member_name(parser, result, i);
                if (!member_names[i])
                    member_names[i] = format_string_arena(parser->arena->arena, "member%d", i);
                member_tys[i] = get_def_type(parser, instruction[2 + i]);
            }
            nominal_type_decl->payload.nom_type.body = record_type(parser->arena, (RecordType) {
                .members = nodes(parser->arena, members_count, member_tys),
                .names = strings(parser->arena, members_count, member_names),
            });
            break;
        }
        case SpvOpTypeRuntimeArray:
        case SpvOpTypeArray: {
            parser->defs[result].type = Typ;
            const Type* element_t = get_def_type(parser, instruction[2]);
            const Node* array_size = NULL;
            if (op != SpvOpTypeRuntimeArray)
                array_size = get_def_ssa_value(parser, instruction[3]);
            parser->defs[result].node = arr_type(parser->arena, (ArrType) {
                .element_type = element_t,
                .size = array_size,
            });
            break;
        }
        case SpvOpTypeVector: {
            parser->defs[result].type = Typ;
            const Type* element_t = get_def_type(parser, instruction[2]);
            parser->defs[result].node = pack_type(parser->arena, (PackType) {
                .element_type = element_t,
                .width = instruction[3],
            });
            break;
        }
        case SpvOpConstant: {
            parser->defs[result].type = Value;
            const Type* t = get_def_type(parser, result_t);
            int width = get_type_bitwidth(t);
            switch (is_type(t)) {
                case Int_TAG: {
                    uint64_t v;
                    if (width == 64) {
                        v = *(uint64_t*)(instruction + 3);
                    } else
                        v = instruction[3];
                    parser->defs[result].node = int_literal(parser->arena, (IntLiteral) {
                        .width = t->payload.int_literal.width,
                        .is_signed = t->payload.int_literal.is_signed,
                        .value = v
                    });
                    break;
                }
                case Float_TAG: {
                    uint64_t v;
                    if (width == 64) {
                        v = *(uint64_t*)(instruction + 3);
                    } else
                        v = instruction[3];
                    parser->defs[result].node = float_literal(parser->arena, (FloatLiteral) {
                        .width = t->payload.float_literal.width,
                        .value = v
                    });
                    break;
                }
                default: error("OpConstant must produce an int or a float");
            }
            break;
        }
        case SpvOpUndef:
        case SpvOpConstantNull: {
            const Type* element_t = get_def_type(parser, result_t);
            parser->defs[result].type = Value;
            parser->defs[result].node = get_default_zero_value(parser->arena, element_t);
            break;
        }
        case SpvOpConstantFalse: {
            parser->defs[result].type = Value;
            parser->defs[result].node = false_lit(parser->arena);
            break;
        }
        case SpvOpConstantTrue: {
            parser->defs[result].type = Value;
            parser->defs[result].node = true_lit(parser->arena);
            break;
        }
        case SpvOpCompositeConstruct:
        case SpvOpConstantComposite: {
            parser->defs[result].type = Value;
            const Type* t = get_def_type(parser, result_t);
            LARRAY(const Node*, contents, size - 3);
            for (size_t i = 0; i < size - 3; i++)
                contents[i] = get_def_ssa_value(parser, instruction[3 + i]);
            parser->defs[result].node = composite_helper(parser->arena, t, nodes(parser->arena, size - 3, contents));
            break;
        }
        case SpvOpVariable: {
            String name = get_name(parser, result);
            name = name ? name : unique_name(parser->arena, "global_variable");

            AddressSpace as = convert_storage_class(instruction[3]);
            const Type* contents_t = get_def_type(parser, result_t);
            AddressSpace as2 = deconstruct_pointer_type(&contents_t);
            assert(as == as2);
            assert(is_data_type(contents_t));

            if (parser->fun) {
                const Node* ptr = first(bind_instruction_outputs_count(parser->current_block.builder, stack_alloc(parser->arena, (StackAlloc) { contents_t }), 1));

                parser->defs[result].type = Value;
                parser->defs[result].node = ptr;

                if (size == 5)
                    bind_instruction_outputs_count(parser->current_block.builder, store(parser->arena, (Store) { ptr, get_def_ssa_value(parser, instruction[4]) }), 1);
            } else {
                Nodes annotations = empty(parser->arena);
                SpvDeco* builtin = find_decoration(parser, result, -1, SpvDecorationBuiltIn);
                if (builtin) {
                    Builtin b = get_builtin_by_spv_id(*builtin->payload.literals.data);
                    assert(b != BuiltinsCount && "Unsupported builtin");
                    annotations = append_nodes(parser->arena, annotations, annotation_value_helper(parser->arena, "Builtin", string_lit_helper(parser->arena, get_builtin_name(b))));
                }

                parser->defs[result].type = Decl;
                Node* global = global_var(parser->mod, annotations, contents_t, name, as);
                parser->defs[result].node = global;

                if (size == 5)
                    global->payload.global_variable.init = get_def_ssa_value(parser, instruction[4]);
            }
            break;
        }
        case SpvOpFunction: {
            assert(parser->defs[result].type == Forward);
            parser->defs[result].type = Decl;
            const Type* t = get_def_type(parser, instruction[4]);
            assert(t && t->tag == FnType_TAG);

            String name = get_name(parser, result);
            if (!name)
                name = unique_name(parser->arena, "function");
            else
                name = unique_name(parser->arena, name);

            Nodes annotations = empty(parser->arena);
            annotations = append_nodes(parser->arena, annotations, annotation(parser->arena, (Annotation) { .name = "Restructure" }));
            SpvDeco* entry_point_type = find_decoration(parser, result, -1, ShdDecorationEntryPointType);
            parser->is_entry_pt = entry_point_type;
            if (entry_point_type) {
                annotations = append_nodes(parser->arena, annotations, annotation_value(parser->arena, (AnnotationValue) {
                    .name = "EntryPoint",
                    .value = string_lit(parser->arena, (StringLiteral) { entry_point_type->payload.str })
                }));

                SpvDeco* entry_point_name = find_decoration(parser, result, -1, ShdDecorationEntryPointName);
                assert(entry_point_name);
                name = entry_point_name->payload.str;

                if (strcmp(entry_point_type->payload.str, "Compute") == 0) {
                    SpvDeco* wg_size_dec = find_decoration(parser, result, -2, SpvExecutionModeLocalSize);
                    assert(wg_size_dec && wg_size_dec->payload.literals.count == 3 && "we require kernels decorated with a workgroup size");
                    annotations = append_nodes(parser->arena, annotations, annotation_values(parser->arena, (AnnotationValues) {
                            .name = "WorkgroupSize",
                            .values = mk_nodes(parser->arena,
                                               int32_literal(parser->arena, wg_size_dec->payload.literals.data[0]),
                                               int32_literal(parser->arena, wg_size_dec->payload.literals.data[1]),
                                               int32_literal(parser->arena, wg_size_dec->payload.literals.data[2]))
                    }));
                } else if (strcmp(entry_point_type->payload.str, "Fragment") == 0) {

                } else if (strcmp(entry_point_type->payload.str, "Vertex") == 0) {

                } else {
                    warn_print("Unknown entry point type '%s' for '%s'\n", entry_point_type->payload.str, name);
                }
            }

            size_t params_count = t->payload.fn_type.param_types.count;
            LARRAY(const Node*, params, params_count);

            for (size_t i = 0; ((parser->words + instruction_offset)[0] & 0xFFFF) == SpvOpFunctionParameter; i++) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                params[i] = get_def_ssa_value(parser, get_result_defined_at(parser, instruction_offset));
                size += s;
                instruction_offset += s;
            }

            Node* fun = function(parser->mod, nodes(parser->arena, params_count, params), name, annotations, t->payload.fn_type.return_types);
            parser->defs[result].node = fun;
            Node* old_fun = parser->fun;
            parser->fun = fun;

            const Node* first_block = NULL;
            while (((parser->words + instruction_offset)[0] & 0xFFFF) != SpvOpFunctionEnd) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                const Node* block = get_definition_by_id(parser, get_result_defined_at(parser, instruction_offset))->node;
                assert(is_basic_block(block));
                if (!first_block)
                    first_block = block;
                size += s;
                instruction_offset += s;
            }

            // Final OpFunctionEnd
            size_t s = parse_spv_instruction_at(parser, instruction_offset);
            size += s;

            // steal the body of the first block, it can't be jumped to anyways!
            if (first_block)
                set_abstraction_body(fun, first_block->payload.basic_block.body);
            parser->fun = old_fun;
            break;
        }
        case SpvOpFunctionEnd: {
            break;
        }
        case SpvOpFunctionParameter: {
            parser->defs[result].type = Value;
            String param_name = get_name(parser, result);
            param_name = param_name ? param_name : format_string_arena(parser->arena->arena, "param%d", parser->fun_arg_i);
            parser->defs[result].node = param(parser->arena, qualified_type_helper(get_def_type(parser, result_t), parser->is_entry_pt), param_name);
            break;
        }
        case SpvOpLabel: {
            struct CurrBlock old = parser->current_block;
            parser->current_block.id = result;

            Nodes params = empty(parser->arena);
            parser->fun_arg_i = 0;
            while (true) {
                SpvOp param_op = (parser->words + instruction_offset)[0] & 0xFFFF;
                bool is_param = false;
                switch (param_op) {
                    case SpvOpPhi: {
                        is_param = true;
                        break;
                    }
                    case SpvOpLine:
                    case SpvOpNoLine:
                        break;
                    default: goto done_with_params;
                }
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                if (is_param) {
                    const Node* param = get_definition_by_id(parser, get_result_defined_at(parser, instruction_offset))->node;
                    assert(param && param->tag == Param_TAG);
                    params = concat_nodes(parser->arena, params, singleton(param));
                }
                size += s;
                instruction_offset += s;
            }

            done_with_params:

            parser->defs[result].type = BB;
            String bb_name = get_name(parser, result);
            bb_name = bb_name ? bb_name : unique_name(parser->arena, "basic_block");
            Node* block = basic_block(parser->arena, params, bb_name);
            parser->defs[result].node = block;

            BodyBuilder* bb = begin_body_with_mem(parser->arena, get_abstraction_mem(block));
            parser->current_block.builder = bb;
            parser->current_block.finished = NULL;
            while (parser->current_block.builder) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                size += s;
                instruction_offset += s;
            }
            assert(parser->current_block.finished);
            set_abstraction_body(block, parser->current_block.finished);
            parser->current_block = old;
            break;
        }
        case SpvOpPhi: {
            parser->defs[result].type = Value;
            String phi_name = get_name(parser, result);
            phi_name = phi_name ? phi_name : unique_name(parser->arena, "phi");
            parser->defs[result].node = param(parser->arena, qualified_type_helper(get_def_type(parser, result_t), false), phi_name);
            assert(size % 2 == 1);
            int num_callsites = (size - 3) / 2;
            for (size_t i = 0; i < num_callsites; i++) {
                SpvId argument_value = instruction[3 + i * 2 + 0];
                SpvId predecessor_block = instruction[3 + i * 2 + 1];

                SpvPhiArgs* new = arena_alloc(parser->decorations_arena, sizeof(SpvPhiArgs));
                *new = (SpvPhiArgs) {
                    .predecessor = predecessor_block,
                    .arg_i = parser->fun_arg_i,
                    .arg = argument_value,
                    .next_ = NULL,
                };

                SpvPhiArgs** found = find_value_dict(SpvId, SpvPhiArgs*, parser->phi_arguments, parser->current_block.id);
                if (found) {
                    SpvPhiArgs* arg = *found;
                    while (arg->next_) {
                        arg = arg->next_;
                    }
                    arg->next_ = new;
                } else {
                    insert_dict(SpvId, SpvPhiArgs*, parser->phi_arguments, parser->current_block.id, new);
                }

                debugv_print("s2s: recorded argument %d (value id=%d) for block %d with predecessor %d\n", parser->fun_arg_i, argument_value, parser->current_block.id, predecessor_block);
            }
            parser->fun_arg_i++;
            break;
        }
        case SpvOpCopyObject: {
            parser->defs[result].type = Value;
            parser->defs[result].node = get_def_ssa_value(parser, instruction[3]);
            break;
        }
        case SpvOpConvertFToU:
        case SpvOpConvertFToS:
        case SpvOpConvertUToF:
        case SpvOpConvertSToF:
        case SpvOpSConvert:
        case SpvOpFConvert:
        case SpvOpUConvert:
        case SpvOpPtrCastToGeneric: {
            const Type* src = get_def_ssa_value(parser, instruction[3]);
            const Type* dst_t = get_def_type(parser, result_t);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = convert_op,
                    .type_arguments = singleton(dst_t),
                    .operands = singleton(src)
            }), 1));
            break;
        }
        case SpvOpConvertPtrToU:
        case SpvOpConvertUToPtr:
        case SpvOpBitcast: {
            const Type* src = get_def_ssa_value(parser, instruction[3]);
            const Type* dst_t = get_def_type(parser, result_t);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = reinterpret_op,
                    .type_arguments = singleton(dst_t),
                    .operands = singleton(src)
            }), 1));
            break;
        }
        case SpvOpInBoundsPtrAccessChain:
        case SpvOpPtrAccessChain:
        case SpvOpInBoundsAccessChain:
        case SpvOpAccessChain: {
            bool has_element = op == SpvOpInBoundsPtrAccessChain || op == SpvOpPtrAccessChain;
            int indices_start = has_element ? 5 : 4;
            int num_indices = size - indices_start;
            LARRAY(const Node*, indices, num_indices);
            const Node* ptr = get_def_ssa_value(parser, instruction[3]);
            const Node* offset = NULL;
            if (has_element)
                offset = get_def_ssa_value(parser, instruction[4]);
            else
                offset = int32_literal(parser->arena, 0);
            for (size_t i = 0; i < num_indices; i++)
                indices[i] = get_def_ssa_value(parser, instruction[indices_start + i]);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, lea(a, (Lea) {
                .ptr = ptr,
                .offset = offset,
                .indices = nodes(a, num_indices, indices),
            }), 1));
            break;
        }
        case SpvOpCompositeExtract: {
            int num_indices = size - 4;
            LARRAY(const Node*, ops, 1 + num_indices);
            ops[0] = get_def_ssa_value(parser, instruction[3]);
            for (size_t i = 0; i < num_indices; i++)
                ops[1 + i] = int32_literal(parser->arena, instruction[4 + i]);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = extract_op,
                    .type_arguments = empty(parser->arena),
                    .operands = nodes(parser->arena, 1 + num_indices, ops)
            }), 1));
            break;
        }
        case SpvOpCompositeInsert: {
            int num_indices = size - 5;
            LARRAY(const Node*, ops, 2 + num_indices);
            ops[0] = get_def_ssa_value(parser, instruction[4]);
            ops[1] = get_def_ssa_value(parser, instruction[3]);
            for (size_t i = 0; i < num_indices; i++)
                ops[2 + i] = int32_literal(parser->arena, instruction[5 + i]);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = insert_op,
                    .type_arguments = empty(parser->arena),
                    .operands = nodes(parser->arena, 2 + num_indices, ops)
            }), 1));
            break;
        }
        case SpvOpVectorShuffle: {
            const Node* src_a = get_def_ssa_value(parser, instruction[3]);
            const Node* src_b = get_def_ssa_value(parser, instruction[4]);

            const Type* src_a_t = get_definition_by_id(parser, instruction[3])->result_type;
            // deconstruct_qualified_type(&src_a_t);
            assert(src_a_t->tag == PackType_TAG);
            size_t num_components_a = src_a_t->payload.pack_type.width;

            int num_components = size - 5;
            LARRAY(const Node*, components, num_components);
            for (size_t i = 0; i < num_components; i++) {
                size_t index = instruction[5 + i];
                const Node* src = src_a;
                if (index >= num_components_a) {
                    index -= num_components_a;
                    src = src_b;
                }
                components[i] = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = extract_op,
                    .type_arguments = empty(parser->arena),
                    .operands = mk_nodes(parser->arena, src, int32_literal(parser->arena, index))
                }), 1));
            }

            parser->defs[result].type = Value;
            parser->defs[result].node = composite_helper(parser->arena, pack_type(parser->arena, (PackType) {
                    .element_type = src_a_t->payload.pack_type.element_type,
                    .width = num_components,
                }), nodes(parser->arena, num_components, components));
            break;
        }
        case SpvOpLoad: {
            const Type* src = get_def_ssa_value(parser, instruction[3]);
            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, load(a, (Load) { src }), 1));
            break;
        }
        case SpvOpStore: {
            const Type* ptr = get_def_ssa_value(parser, instruction[1]);
            const Type* value = get_def_ssa_value(parser, instruction[2]);
            bind_instruction_outputs_count(parser->current_block.builder, store(a, (Store) { ptr, value }), 0);
            break;
        }
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized: {
            const Node* dst = get_def_ssa_value(parser, instruction[1]);
            const Node* src = get_def_ssa_value(parser, instruction[2]);
            const Node* cnt;
            if (op == SpvOpCopyMemory) {
                const Type* elem_t = src->type;
                deconstruct_qualified_type(&elem_t);
                deconstruct_pointer_type(&elem_t);
                cnt = first(bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                    .op = size_of_op,
                    .type_arguments = singleton(elem_t),
                    .operands = empty(parser->arena)
                }), 1));
            } else {
                cnt = get_def_ssa_value(parser, instruction[3]);
            }
            bind_instruction_outputs_count(parser->current_block.builder, copy_bytes(parser->arena, (CopyBytes) {
                .src = src,
                .dst = dst,
                .count = cnt,
            }), 0);
            break;
        }
        case SpvOpSelectionMerge:
            break;
        case SpvOpLifetimeStart:
        case SpvOpLifetimeStop:
            // these are no-ops ... I think ?
            break;
        case SpvOpFunctionCall: {
            parser->defs[result].type = Value;
            const Node* callee = get_def_decl(parser, instruction[3]);
            size_t num_args = size - 4;
            LARRAY(const Node*, args, num_args);
            for (size_t i = 0; i < num_args; i++)
                args[i] = get_def_ssa_value(parser, instruction[4 + i]);

            int rslts_count = get_def_type(parser, result_t) == unit_type(parser->arena) ? 0 : 1;

            if (callee->tag == Function_TAG) {
                const Node* fn = callee; //callee->payload.fn_addr.fn;
                String fn_name = get_abstraction_name(fn);
                if (string_starts_with(fn_name, "__shady")) {
                    char* copy = malloc(strlen(fn_name) + 1);
                    memcpy(copy, fn_name, strlen(fn_name) + 1);
                    strtok(copy, ":");
                    char* intrinsic = strtok(NULL, ":");
                    assert(strcmp(intrinsic, "prim_op") == 0);
                    char* primop = strtok(NULL, ":");
                    Op op = PRIMOPS_COUNT;
                    for (size_t i = 0; i < PRIMOPS_COUNT; i++) {
                        if (strcmp(get_primop_name(i), primop) == 0) {
                            op = i;
                            break;
                        }
                    }
                    assert(op != PRIMOPS_COUNT);
                    //assert(false && intrinsic);
                    Nodes rslts = bind_instruction_outputs_count(parser->current_block.builder, prim_op(parser->arena, (PrimOp) {
                        .op = op,
                        .type_arguments = empty(parser->arena),
                        .operands = nodes(parser->arena, num_args, args)
                    }), rslts_count);

                    if (rslts_count == 1)
                        parser->defs[result].node = first(rslts);

                    break;
                }
            }
            Nodes rslts = bind_instruction_outputs_count(parser->current_block.builder, call(parser->arena, (Call) {
                .callee = fn_addr_helper(parser->arena, callee),
                .args = nodes(parser->arena, num_args, args)
            }), rslts_count);

            if (rslts_count == 1)
                parser->defs[result].node = first(rslts);

            break;
        }
        case SpvOpExtInst: {
            String set = get_def_string(parser, instruction[3]);
            assert(set);
            uint32_t ext_instr = instruction[4];
            size_t num_args = size - 5;
            LARRAY(const Node*, args, num_args);
            for (size_t i = 0; i < num_args; i++)
                args[i] = get_def_ssa_value(parser, instruction[5 + i]);

            const Node* instr = NULL;
            if (strcmp(set, "OpenCL.std") == 0) {
                switch (ext_instr) {
                    case OpenCLstd_Mad:
                        assert(num_args == 3);
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = mul_op,
                            .operands = mk_nodes(parser->arena, args[0], args[1])
                        });
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = add_op,
                            .operands = mk_nodes(parser->arena, instr, args[2])
                        });
                        break;
                    case OpenCLstd_Floor:
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = floor_op,
                        .operands = singleton(args[0])
                        });
                        break;
                    case OpenCLstd_Sqrt:
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = sqrt_op,
                            .operands = singleton(args[0])
                        });
                        break;
                    case OpenCLstd_Fabs:
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = abs_op,
                            .operands = singleton(args[0])
                        });
                        break;
                    case OpenCLstd_Sin:
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = sin_op,
                            .operands = singleton(args[0])
                        });
                        break;
                    case OpenCLstd_Cos:
                        instr = prim_op(parser->arena, (PrimOp) {
                            .op = cos_op,
                            .operands = singleton(args[0])
                        });
                        break;
                    default: error("unhandled extended instruction %d in set '%s'", ext_instr, set);
                }
            } else if (strcmp(set, "GLSL.std.450") == 0) {
                switch (ext_instr) {
                    case GLSLstd450Fma:
                        assert(num_args == 3);
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = mul_op,
                                .operands = mk_nodes(parser->arena, args[0], args[1])
                        });
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = add_op,
                                .operands = mk_nodes(parser->arena, instr, args[2])
                        });
                        break;
                    case GLSLstd450Floor:
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = floor_op,
                                .operands = singleton(args[0])
                        });
                        break;
                    case GLSLstd450Sqrt:
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = sqrt_op,
                                .operands = singleton(args[0])
                        });
                        break;
                    case GLSLstd450FAbs:
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = abs_op,
                                .operands = singleton(args[0])
                        });
                        break;
                    case GLSLstd450Sin:
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = sin_op,
                                .operands = singleton(args[0])
                        });
                        break;
                    case GLSLstd450Cos:
                        instr = prim_op(parser->arena, (PrimOp) {
                                .op = cos_op,
                                .operands = singleton(args[0])
                        });
                        break;
                    case GLSLstd450FMin: instr = prim_op(parser->arena, (PrimOp) { .op = min_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450SMin: instr = prim_op(parser->arena, (PrimOp) { .op = min_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450UMin: instr = prim_op(parser->arena, (PrimOp) { .op = min_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450FMax: instr = prim_op(parser->arena, (PrimOp) { .op = max_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450SMax: instr = prim_op(parser->arena, (PrimOp) { .op = max_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450UMax: instr = prim_op(parser->arena, (PrimOp) { .op = max_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    case GLSLstd450Exp: instr = prim_op(parser->arena, (PrimOp) { .op = exp_op, .operands = singleton(args[0]) }); break;
                    case GLSLstd450Pow: instr = prim_op(parser->arena, (PrimOp) { .op = pow_op, .operands = mk_nodes(parser->arena, args[0], args[1]) }); break;
                    default: error("unhandled extended instruction %d in set '%s'", ext_instr, set);
                }
            } else {
                error("Unknown extended instruction set '%s'", set);
            }

            parser->defs[result].type = Value;
            parser->defs[result].node = first(bind_instruction_outputs_count(parser->current_block.builder, instr, 1));
            break;
        }
        case SpvOpBranch: {
            BodyBuilder* bb = parser->current_block.builder;
            parser->current_block.finished = finish_body(bb, jump(parser->arena, (Jump) {
                .target = get_def_block(parser, instruction[1]),
                .args = get_args_from_phi(parser, instruction[1], parser->current_block.id),
                .mem = bb_mem(bb)
            }));
            parser->current_block.builder = NULL;
            break;
        }
        case SpvOpBranchConditional: {
            SpvId destinations[2] = { instruction[2], instruction[3] };
            BodyBuilder* bb = parser->current_block.builder;
            parser->current_block.finished = finish_body(bb, branch(parser->arena, (Branch) {
                .true_jump = jump_helper(parser->arena, get_def_block(parser, destinations[0]), get_args_from_phi(parser, destinations[0], parser->current_block.id), bb_mem(bb)),
                .false_jump = jump_helper(parser->arena, get_def_block(parser, destinations[1]), get_args_from_phi(parser, destinations[1], parser->current_block.id), bb_mem(bb)),
                .condition = get_def_ssa_value(parser, instruction[1]),
            }));
            parser->current_block.builder = NULL;
            break;
        }
        case SpvOpReturn:
        case SpvOpReturnValue: {
            Nodes args;
            if (op == SpvOpReturn)
                args = empty(parser->arena);
            else
                args = singleton(get_def_ssa_value(parser, instruction[1]));
            BodyBuilder* bb = parser->current_block.builder;
            parser->current_block.finished = finish_body(bb, fn_ret(parser->arena, (Return) {
                .args = args,
            }));
            parser->current_block.builder = NULL;
            break;
        }
        default: error("Unsupported op: %d, size: %d", op, size);
    }

    if (has_result) {
        parser->defs[result].final_size = size;
    }

    return size;
}

SpvDef* get_definition_by_id(SpvParser* parser, size_t id) {
    assert(id > 0 && id < parser->header.bound);
    if (parser->defs[id].type == Nothing)
    error("there is no Op that defines result %zu", id);
    if (parser->defs[id].type == Forward)
        parse_spv_instruction_at(parser, parser->defs[id].instruction_offset);
    assert(parser->defs[id].type != Forward);
    return &parser->defs[id];
}

KeyHash hash_spvid(SpvId* p) {
    return hash_murmur(p, sizeof(SpvId));
}

bool compare_spvid(SpvId* pa, SpvId* pb) {
    if (pa == pb) return true;
    if (!pa || !pb) return false;
    return *pa == *pb;
}

S2SError parse_spirv_into_shady(const CompilerConfig* config, size_t len, const char* data, String name, Module** dst) {
    ArenaConfig aconfig = default_arena_config(&config->target);
    IrArena* a = new_ir_arena(&aconfig);
    *dst = new_module(a, name);

    SpvParser parser = {
        .cursor = 0,
        .len = len / sizeof(uint32_t),
        .words = (uint32_t*) data,
        .mod = *dst,
        .arena = get_module_arena(*dst),

        .decorations_arena = new_arena(),
        .phi_arguments = new_dict(SpvId, SpvPhiArgs*, (HashFn) hash_spvid, (CmpFn) compare_spvid),
    };

    if (!parse_spv_header(&parser))
        return S2S_FailedParsingGeneric;
    assert(parser.header.bound > 0 && parser.header.bound < 512 * 1024 * 1024); // sanity check
    parser.defs = calloc(parser.header.bound, sizeof(SpvDef));

    scan_definitions(&parser);

    while (parser.cursor < parser.len) {
        parser.cursor += parse_spv_instruction_at(&parser, parser.cursor);
    }

    destroy_dict(parser.phi_arguments);
    destroy_arena(parser.decorations_arena);
    free(parser.defs);

    return S2S_Success;
}
