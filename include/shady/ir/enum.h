#ifndef SHADY_IR_ENUM_H
#define SHADY_IR_ENUM_H

typedef enum {
    ShdIntSize8,
    ShdIntSize16,
    ShdIntSize32,
    ShdIntSize64,
} ShdIntSize;

enum {
    ShdIntSizeMin = ShdIntSize8,
    ShdIntSizeMax = ShdIntSize64,
};

typedef enum {
    ShdFloatFormat16,
    ShdFloatFormat32,
    ShdFloatFormat64
} ShdFloatFormat;

typedef enum {
    ShdStructFlagNone,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    ShdStructFlagBlock = 1,
    /// Will emit explicit layout annotations
    ShdStructFlagExplicitLayout = 2,
} ShdStructFlags;

typedef enum {
    ShdArrayFlagNone,
    ShdArrayFlagExplicitLayout = 1,
} ShdArrayFlags;

typedef enum {
    ShdScopeTop,
    ShdScopeCrossDevice = ShdScopeTop,
    ShdScopeDevice,
    ShdScopeWorkgroup,
    ShdScopeSubgroup,
    ShdScopeInvocation,
    ShdScopeBottom = ShdScopeInvocation
} ShdScope;

String shd_get_scope_name(ShdScope);

// see enum.json
#include "enum_generated.h"

#endif
