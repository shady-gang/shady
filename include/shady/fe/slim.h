#ifndef SHADY_SLIM_H
#define SHADY_SLIM_H

#include "shady/ir/module.h"
#include "shady/config.h"

typedef enum {
    SlimOpDereference,
    SlimOpAssign,
    SlimOpAddrOf,
    SlimOpSubscript,
    SlimOpBindVal,
    SlimOpBindVar,
    SlimOpBindContinuations,
    SlimOpUnbound,
} SlimFrontEndOpCodes;

typedef struct {
    bool front_end;
} SlimParserConfig;

Module* shd_parse_slim_module(const CompilerConfig* config, const SlimParserConfig* pconfig, const char* contents, String name);

#endif
