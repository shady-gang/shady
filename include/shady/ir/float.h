#ifndef SHADY_IR_FLOAT_H
#define SHADY_IR_FLOAT_H

#include "shady/ir/grammar.h"

static inline int float_size_in_bytes(ShdFloatFormat s) {
    switch (s) {
        case ShdFloatFormat16: return 2;
        case ShdFloatFormat32: return 4;
        case ShdFloatFormat64: return 8;
    }
}

const Type* shd_fp16_type(IrArena* arena);
const Type* shd_fp32_type(IrArena* arena);
const Type* shd_fp64_type(IrArena* arena);

const Node* shd_fp_literal_helper(IrArena* a, ShdFloatFormat size, double value);

const Node* shd_float32_literal(IrArena* arena, double f);
const Node* shd_float64_literal(IrArena* arena, double f);

const FloatLiteral* shd_resolve_to_float_literal(const Node* node);
double shd_get_float_literal_value(FloatLiteral literal);

#endif
