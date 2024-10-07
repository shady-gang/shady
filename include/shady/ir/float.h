#ifndef SHADY_IR_FLOAT_H
#define SHADY_IR_FLOAT_H

#include "shady/ir/base.h"

typedef enum {
    FloatTy16,
    FloatTy32,
    FloatTy64
} FloatSizes;

static inline int float_size_in_bytes(FloatSizes s) {
    switch (s) {
        case FloatTy16: return 2;
        case FloatTy32: return 4;
        case FloatTy64: return 8;
    }
}

const Type* shd_fp16_type(IrArena* arena);
const Type* shd_fp32_type(IrArena* arena);
const Type* shd_fp64_type(IrArena* arena);

const Node* shd_fp_literal_helper(IrArena* a, FloatSizes size, double value);

#endif
