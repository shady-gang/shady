#ifndef SHADY_S2S
#define SHADY_S2S

#include "shady/ir.h"

typedef enum {
    S2S_Success,
    S2S_FailedParsingGeneric,
} S2SError;

S2SError parse_spirv_into_shady(const CompilerConfig* config, size_t len, const char* data, String name, Module**);

#endif
