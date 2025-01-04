#ifndef SHADY_IR_ENUM_H
#define SHADY_IR_ENUM_H

typedef enum {
    IntTy8,
    IntTy16,
    IntTy32,
    IntTy64,
} IntSizes;

enum {
    IntSizeMin = IntTy8,
    IntSizeMax = IntTy64,
};

typedef enum {
    FloatTy16,
    FloatTy32,
    FloatTy64
} FloatSizes;

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

typedef enum {
    ShdScopeTop,
    ShdScopeCrossDevice = ShdScopeTop,
    ShdScopeDevice,
    ShdScopeWorkgroup,
    ShdScopeSubgroup,
    ShdScopeInvocation,
    ShdScopeBottom = ShdScopeInvocation
} ShdScope;

#endif
