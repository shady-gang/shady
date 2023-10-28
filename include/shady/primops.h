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

#define STACK_PRIMOPS(P) \
P(1, push_stack)                \
P(1, pop_stack)                 \
P(0, get_stack_pointer)         \
P(0, get_stack_base)            \
P(1, set_stack_pointer)         \

#define MEMORY_PRIMOPS(P) \
P(1, alloca)              \
P(1, alloca_logical)      \
P(1, alloca_subgroup)     \
P(0, load)                \
P(1, store)               \
P(0, lea)                 \
P(1, memcpy)              \

#define LAYOUT_PRIMOPS(P) \
P(0, size_of)             \
P(0, align_of)            \
P(0, offset_of)           \

#define SUBGROUP_INTRINSICS_PRIMOPS(P) \
P(0, subgroup_elect_first)             \
P(0, subgroup_broadcast_first)         \
P(0, subgroup_assume_uniform)          \
P(0, subgroup_reduce_sum)              \
P(0, subgroup_active_mask)             \
P(0, subgroup_ballot)                  \

#define SYNTAX_SUGAR_OPS(P) \
P(1, assign)                \
P(1, subscript)             \
P(1, deref)                 \

#define PRIMOPS(P)              \
P(0, quote)                     \
ARITHM_PRIMOPS(P)               \
BITSTUFF_PRIMOPS(P)             \
CMP_PRIMOPS(P)                  \
MATH_PRIMOPS(P)                 \
SHIFT_PRIMOPS(P)                \
MEMORY_PRIMOPS(P)               \
LAYOUT_PRIMOPS(P)               \
P(0, select)                    \
P(0, convert)                   \
P(0, reinterpret)               \
P(0, extract)                   \
P(0, extract_dynamic)           \
P(0, insert)                    \
P(0, shuffle)                   \
P(1, debug_printf)              \
P(1, sample_texture)            \
SUBGROUP_INTRINSICS_PRIMOPS(P)  \
/* these are all lowered away */\
STACK_PRIMOPS(P)                \
SYNTAX_SUGAR_OPS(P)             \
P(1, create_joint_point)        \
P(1, default_join_point)        \
P(0, empty_mask)                \
P(0, mask_is_thread_active)     \

typedef enum Op_ {
#define DECLARE_PRIMOP_ENUM(has_side_effects, name) name##_op,
PRIMOPS(DECLARE_PRIMOP_ENUM)
#undef DECLARE_PRIMOP_ENUM
    PRIMOPS_COUNT
} Op;

extern const char* primop_names[];
bool has_primop_got_side_effects(Op op);
