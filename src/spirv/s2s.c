#include "s2s.h"

// this avoids polluting the namespace
#define SpvHasResultAndType ShadySpvHasResultAndType

// hacky nonsense because this guy is not declared static
typedef enum SpvOp_ SpvOp;
static void SpvHasResultAndType(SpvOp opcode, bool *hasResult, bool *hasResultType);

#define SPV_ENABLE_UTILITY_CODE 1
#include "spirv/unified1/spirv.h"

// TODO: reserve real decoration IDs
typedef enum {
    ShdDecorationName = 999999,
    ShdDecorationMemberName = 999998,
} ShdDecoration;

#include "log.h"
#include "arena.h"
#include "portability.h"

#include "../shady/type.h"

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

typedef struct {
    size_t cursor;
    size_t len;
    uint32_t* words;
    Module* mod;
    IrArena* arena;

    Node* fun;
    BodyBuilder* bb;
    const Node* finished_body;

    SpvHeader header;
    SpvDef* defs;
    Arena* decorations_arena;
} SpvParser;

SpvDef* get_definition_by_id(SpvParser* parser, size_t id);

SpvDef* new_def(SpvParser* parser) {
    SpvDef* interned = arena_alloc(parser->decorations_arena, sizeof(SpvDef));
    SpvDef empty = {};
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

const Type* get_def_type(SpvParser* parser, SpvId id) {
    assert(parser->defs[id].type == Typ);
    const Type* t = parser->defs[id].node;
    assert(t && is_type(t));
    return t;
}

const Type* get_def_ssa_value(SpvParser* parser, SpvId id) {
    assert(parser->defs[id].type == Value);
    const Node* n = parser->defs[id].node;
    assert(n && (is_value(n) || is_instruction(n)));
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
        case SpvStorageClassWorkgroup:             return AsSharedPhysical;
        case SpvStorageClassCrossWorkgroup:        return AsGlobalPhysical;
        case SpvStorageClassPhysicalStorageBuffer: return AsGlobalPhysical;
        case SpvStorageClassPrivate:               return AsPrivatePhysical;
        case SpvStorageClassFunction:              return AsPrivatePhysical;
        case SpvStorageClassGeneric:               return AsGeneric;
        case SpvStorageClassPushConstant:
        case SpvStorageClassAtomicCounter:
        case SpvStorageClassImage:
        case SpvStorageClassStorageBuffer:
            error("TODO");
        case SpvStorageClassUniformConstant:
        case SpvStorageClassUniform:
            error("TODO");
        case SpvStorageClassCallableDataKHR:
        case SpvStorageClassIncomingCallableDataKHR:
        case SpvStorageClassRayPayloadKHR:
        case SpvStorageClassHitAttributeKHR:
        case SpvStorageClassIncomingRayPayloadKHR:
        case SpvStorageClassShaderRecordBufferKHR:
            error("Unsupported");
        case SpvStorageClassCodeSectionINTEL:
        case SpvStorageClassDeviceOnlyINTEL:
        case SpvStorageClassHostOnlyINTEL:
        case SpvStorageClassMax:
            error("Unsupported");
    }
}

typedef struct {
    bool valid;
    Op op;
    int base_size;
} SpvShdOpMapping;

static SpvShdOpMapping spv_shd_op_mapping[] = {
    [SpvOpSLessThan] = { 1, lt_op, 3 }
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

size_t parse_spv_instruction_at(SpvParser* parser, size_t instruction_offset) {
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
        }
    }

    instruction_offset += size;

    if (convert_spv_op(op)) {
        assert(parser->bb);
        SpvShdOpMapping shd_op = *convert_spv_op(op);
        int num_ops = size - shd_op.base_size;
        LARRAY(const Node*, ops, num_ops);
        for (size_t i = 0; i < num_ops; i++)
            ops[i] = get_def_ssa_value(parser, instruction[shd_op.base_size + i]);
        int results_count = has_result ? 1 : 0;
        Nodes results = bind_instruction_extra(parser->bb, prim_op(parser->arena, (PrimOp) {
            .op = shd_op.op,
            .type_arguments = empty(parser->arena),
            .operands = nodes(parser->arena, num_ops, ops)
        }), results_count, NULL, NULL);
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
        case SpvOpName:
        case SpvOpMemberName: {
            SpvId target = instruction[1];
            ShdDecoration decoration = op == SpvOpName ? ShdDecorationName : ShdDecorationMemberName;
            int name_offset = op == SpvOpName ? 3 : 4;
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
        case SpvOpDecorate:
        case SpvOpMemberDecorate: {
            SpvDef payload = { Literals };
            SpvId target = instruction[1];
            int data_offset = op == SpvOpMemberDecorate ? 4 : 3;
            payload.literals.count = size - data_offset;
            payload.literals.data = instruction + data_offset;
            SpvDeco deco = {
                    .payload = payload,
                    .member = op == SpvOpMemberDecorate ? (int) instruction[3] : -1,
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
                .return_types = singleton(return_t),
                .param_types = nodes(parser->arena, size - 3, param_ts)
            });
            break;
        }
        case SpvOpConstant: {
            parser->defs[result].type = Value;
            const Type* t = get_def_type(parser, result_t);
            int width = get_type_bitwidth(t);
            switch (is_type(t)) {
                case Int_TAG: {
                    IntLiteralValue v;
                    if (width == 64) {
                        v.u64 = *(uint64_t*)(instruction + 3);
                    } else
                        v.u64 = instruction[3];
                    parser->defs[result].node = int_literal(parser->arena, (IntLiteral) {
                        .width = t->payload.int_literal.width,
                        .is_signed = t->payload.int_literal.is_signed,
                        .value = v
                    });
                    break;
                }
                case Float_TAG: {
                    FloatLiteralValue v;
                    if (width == 64) {
                        v.b64 = *(uint64_t*)(instruction + 3);
                    } else
                        v.b64 = instruction[3];
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
        case SpvOpFunction: {
            const Type* t = get_def_type(parser, instruction[4]);
            assert(t && t->tag == FnType_TAG);

            size_t params_count = t->payload.fn_type.param_types.count;
            LARRAY(const Node*, params, params_count);

            for (size_t i = 0; ((parser->words + instruction_offset)[0] & 0xFFFF) == SpvOpFunctionParameter; i++) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                params[i] = get_def_ssa_value(parser, get_result_defined_at(parser, instruction_offset));
                size += s;
                instruction_offset += s;
            }

            Node* fun = function(parser->mod, nodes(parser->arena, params_count, params), get_name(parser, result), empty(parser->arena), t->payload.fn_type.return_types);
            parser->defs[result].type = Decl;
            parser->defs[result].node = fun;
            Node* old_fun = parser->fun;
            parser->fun = fun;

            const Node* first_block = NULL;
            while (((parser->words + instruction_offset)[0] & 0xFFFF) != SpvOpFunctionEnd) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                const Node* block = get_definition_by_id(parser, get_result_defined_at(parser, instruction_offset))->node;
                assert(is_terminator(block));
                if (!first_block)
                    first_block = block;
                size += s;
                instruction_offset += s;
            }

            // Final OpFunctionEnd
            size_t s = parse_spv_instruction_at(parser, instruction_offset);
            size += s;

            fun->payload.fun.body = first_block;
            parser->fun = old_fun;
            break;
        }
        case SpvOpFunctionParameter: {
            parser->defs[result].type = Value;
            parser->defs[result].node = var(parser->arena, get_def_type(parser, result_t), get_name(parser, result));
            break;
        }
        case SpvOpLabel: {
            Nodes params = empty(parser->arena);
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
                    assert(param && param->tag == Variable_TAG);
                    params = concat_nodes(parser->arena, params, singleton(param));
                }
                size += s;
                instruction_offset += s;
            }

            done_with_params:

            parser->defs[result].type = BB;
            String bb_name = get_name(parser, result);
            bb_name = bb_name ? bb_name : unique_name(parser->arena, "basic_block");
            parser->defs[result].node = basic_block(parser->arena, parser->fun, params, bb_name);

            assert(!parser->bb);
            BodyBuilder* bb = begin_body(parser->mod);
            parser->bb = bb;
            while (parser->bb) {
                size_t s = parse_spv_instruction_at(parser, instruction_offset);
                assert(s > 0);
                size += s;
                instruction_offset += s;
            }
        }
        case SpvOpPhi: {
            error("TODO: OpPhi")
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

S2SError parse_spirv_into_shady(Module* dst, size_t len, uint32_t* words) {
    SpvParser parser = {
        .cursor = 0,
        .len = len / sizeof(uint32_t),
        .words = words,
        .mod = dst,
        .arena = get_module_arena(dst),

        .decorations_arena = new_arena(),
    };

    if (!parse_spv_header(&parser))
        return S2S_FailedParsingGeneric;
    assert(parser.header.bound > 0 && parser.header.bound < 512 * 1024 * 1024); // sanity check
    parser.defs = calloc(parser.header.bound, sizeof(SpvDef));

    scan_definitions(&parser);

    while (parser.cursor < parser.len) {
        parser.cursor += parse_spv_instruction_at(&parser, parser.cursor);
    }

    destroy_arena(parser.decorations_arena);
    free(parser.defs);

    return S2S_Success;
}
