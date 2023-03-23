#include "s2s.h"

// this avoids polluting the namespace
#define SpvHasResultAndType ShadySpvHasResultAndType

// hacky nonsense because this guy is not declared static
typedef enum SpvOp_ SpvOp;
static void SpvHasResultAndType(SpvOp opcode, bool *hasResult, bool *hasResultType);

#define SPV_ENABLE_UTILITY_CODE 1
#include "spirv/unified1/spirv.h"

#include "log.h"

#include <assert.h>
#include <stdlib.h>

typedef struct {
    struct {
        uint8_t major, minor;
    } version;
    uint32_t generator;
    uint32_t bound;
} SpvHeader;

typedef struct {
    enum { Todo, Str, Decl, Value } type;
    union {
        Node* node;
        String str;
    };
} SpvDef;

typedef struct {
    size_t cursor;
    size_t len;
    uint32_t* words;
    Module* mod;

    SpvHeader header;
    SpvDef* defs;
} SpvParser;

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
    };

    if (!parse_spv_header(&parser))
        return S2S_FailedParsingGeneric;
    assert(parser.header.bound > 0 && parser.header.bound < 512 * 1024 * 1024); // sanity check
    parser.defs = calloc(parser.header.bound, sizeof(SpvDef));

    while (parser.cursor < parser.len) {
        parse_spv_instruction(&parser);
    }

    return S2S_Success;
}
