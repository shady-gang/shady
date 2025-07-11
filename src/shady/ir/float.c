#include "shady/ir/float.h"

#include "shady/analysis/literal.h"

#include "log.h"

#include <assert.h>

const Type* shd_fp16_type(IrArena* arena) { return float_type(arena, (Float) { .width = ShdFloatFormat16 }); }
const Type* shd_fp32_type(IrArena* arena) { return float_type(arena, (Float) { .width = ShdFloatFormat32 }); }
const Type* shd_fp64_type(IrArena* arena) { return float_type(arena, (Float) { .width = ShdFloatFormat64 }); }

const Node* shd_fp_literal_helper(IrArena* a, ShdFloatFormat size, double value) {
    switch (size) {
        case ShdFloatFormat16: assert(false); break;
        case ShdFloatFormat32: {
            float f = value;
            uint64_t bits = 0;
            memcpy(&bits, &f, sizeof(f));
            return float_literal(a, (FloatLiteral) { .width = size, .value = bits });
        }
        case ShdFloatFormat64: {
            uint64_t bits = 0;
            memcpy(&bits, &value, sizeof(value));
            return float_literal(a, (FloatLiteral) { .width = size, .value = bits });
        }
    }
}

const FloatLiteral* shd_resolve_to_float_literal(const Node* node) {
    node = shd_resolve_node_to_definition(node, shd_default_node_resolve_config());
    if (!node)
        return NULL;
    if (node->tag == FloatLiteral_TAG)
        return &node->payload.float_literal;
    return NULL;
}

static_assert(sizeof(float) == sizeof(uint64_t) / 2, "floats aren't the size we expect");
double shd_get_float_literal_value(FloatLiteral literal) {
    double r;
    switch (literal.width) {
        case ShdFloatFormat16:
            shd_error_print("TODO: fp16 literals");
        shd_error_die();
        SHADY_UNREACHABLE;
        break;
        case ShdFloatFormat32: {
            float f;
            memcpy(&f, &literal.value, sizeof(float));
            r = (double) f;
            break;
        }
        case ShdFloatFormat64:
            memcpy(&r, &literal.value, sizeof(double));
        break;
    }
    return r;
}

const Node* shd_float32_literal(IrArena* arena, double d) {
    float f = d;
    uint64_t u = 0;
    memcpy(&u, &f, sizeof(f));
    return float_literal_helper(arena, ShdFloatFormat32, u);
}

const Node* shd_float64_literal(IrArena* arena, double d) {
    uint64_t u = 0;
    memcpy(&u, &d, sizeof(d));
    return float_literal_helper(arena, ShdFloatFormat64, u);
}
