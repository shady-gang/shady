#ifndef SHADY_S2S
#define SHADY_S2S

#include "shady/ir.h"

typedef enum {
    S2S_Success,
    S2S_FailedParsingGeneric,
} S2SError;

typedef struct CompilerConfig_ CompilerConfig;
S2SError shd_parse_spirv(const CompilerConfig* config, size_t len, const char* data, String name, Module** dst);

#endif
