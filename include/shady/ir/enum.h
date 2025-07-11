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
    ShdRecordFlagNone,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    ShdRecordFlagMultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    ShdRecordFlagBlock
} ShdRecordFlags;

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
