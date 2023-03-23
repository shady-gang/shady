#include "s2s.h"

// this avoids polluting the namespace
#define SpvHasResultAndType ShadySpvHasResultAndType

// hacky nonsense because this guy is not declared static
typedef enum SpvOp_ SpvOp;
static void SpvHasResultAndType(SpvOp opcode, bool *hasResult, bool *hasResultType);

#define SPV_ENABLE_UTILITY_CODE 1
#include "spirv/unified1/spirv.h"

#include "log.h"
#include "arena.h"

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
    enum { Todo, Str, Typ, Decl, Value, Literals } type;
    union {
        Node* node;
        String str;
        struct { size_t count; uint32_t* data; } literals;
    };
    SpvDeco* decoration;
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

    SpvHeader header;
    SpvDef* defs;
    Arena* decorations_arena;
} SpvParser;

SpvDef* new_def(SpvParser* parser) {
    SpvDef* interned = arena_alloc(parser->decorations_arena, sizeof(SpvDef));
    SpvDef empty = {};
    memcpy(interned, &empty, sizeof(SpvDef));
    return interned;
}

void add_decoration(SpvParser* parser, SpvId id, SpvDeco decoration) {
    SpvDef* tgt_def = &parser->defs[id];
    while (tgt_def->decoration) {
        tgt_def = &tgt_def->decoration->payload;
    }
    SpvDeco* interned = arena_alloc(parser->decorations_arena, sizeof(SpvDeco));
    memcpy(interned, &decoration, sizeof(SpvDeco));
    tgt_def->decoration = interned;
}

SpvDeco* find_decoration(SpvParser* parser, SpvId id, int member, SpvDecoration tag) {
    SpvDef* tgt_def = &parser->defs[id];
    while (tgt_def->decoration) {
        if (tgt_def->decoration->decoration == tag && (member < 0 || tgt_def->decoration->member == member))
            return tgt_def->decoration;
        tgt_def = &tgt_def->decoration->payload;
    }
    return NULL;
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

bool parse_spv_instruction(SpvParser* parser) {
    size_t available = parser->len - parser->cursor;
    assert(available >= 1);
    uint32_t* instruction = parser->words + parser->cursor;
    SpvOp op = instruction[0] & 0xFFFF;
    int size = (int) ((instruction[0] >> 16u) & 0xFFFFu);

    SpvId result = 0;
    bool has_result, has_type;
    SpvHasResultAndType(op, &has_result, &has_type);
    if (has_result)
        result = instruction[1];

    switch (op) {
        // shady doesn't care, the emitter will set those up
        case SpvOpMemoryModel:
        case SpvOpCapability: break;
        // these are basically just strings and we can infer how to handle them from ctx at the uses
        case SpvOpName:
        case SpvOpExtInstImport:
        case SpvOpSource:
        case SpvOpString: {
            parser->defs[result].type = Str;
            parser->defs[result].str = decode_spv_string_literal(parser, instruction + 2);
            break;
        }
        // more debug nonsense
        case SpvOpModuleProcessed:
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
        default: error("Unsupported op: %d, size: %d", op, size);
    }

    parser->cursor += size;
    return true;
}

S2SError parse_spirv_into_shady(Module* dst, size_t len, uint32_t* words) {
    SpvParser parser = {
        .cursor = 0,
        .len = len,
        .words = words,
        .mod = dst,
        .arena = get_module_arena(dst),

        .decorations_arena = new_arena(),
    };

    if (!parse_spv_header(&parser))
        return S2S_FailedParsingGeneric;
    assert(parser.header.bound > 0 && parser.header.bound < 512 * 1024 * 1024); // sanity check
    parser.defs = calloc(parser.header.bound, sizeof(SpvDef));

    while (parser.cursor < parser.len) {
        parse_spv_instruction(&parser);
    }

    destroy_arena(parser.decorations_arena);

    return S2S_Success;
}
