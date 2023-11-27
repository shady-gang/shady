#ifndef SHADY_IR_H
#error "do not include this file by itself, include shady/ir.h instead"
#endif

// All the primops are defined in this file, using the following syntax:
// PRIMOP(has_side_effects, name)

#define ARITHM_PRIMOPS(P) \
P(0, add)                 \
P(0, add_carry)           \
P(0, sub)                 \
P(0, sub_borrow)          \
P(0, mul)                 \
P(0, mul_extended)        \
P(0, div)                 \
P(0, mod)                 \
P(0, neg)                 \

#define BITSTUFF_PRIMOPS(P) \
P(0, not)                   \
P(0, and)                   \
P(0, or)                    \
P(0, xor)                   \

#define CMP_PRIMOPS(P) \
P(0, gt)               \
P(0, gte)              \
P(0, lt)               \
P(0, lte)              \
P(0, eq)               \
P(0, neq)              \

#define SHIFT_PRIMOPS(P) \
P(0, rshift_logical)     \
P(0, rshift_arithm)      \
P(0, lshift)             \

#define MATH_PRIMOPS(P) \
P(0, sqrt)              \
P(0, inv_sqrt)          \
P(0, pow)               \
P(0, exp)               \
P(0, floor)             \
P(0, ceil)              \
P(0, round)             \
P(0, fract)             \
P(0, min)               \
P(0, max)               \
P(0, abs)               \
P(0, sign)              \
P(0, sin)               \
P(0, cos)               \

#include "primops_generated.h"

String get_primop_name(Op op);
bool has_primop_got_side_effects(Op op);
