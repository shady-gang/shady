#include "s2s.h"
#include "spirv/unified1/spirv.h"

#include <assert.h>

typedef struct {
    struct {
        uint8_t major, minor;
    } version;
    uint32_t generator;
    uint32_t bound;
} SpvHeader;

typedef struct {
    size_t cursor;
    size_t len;
    uint32_t* words;
    Module* mod;

    SpvHeader header;
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

    assert(false && "TODO");

    return S2S_Success;
}
