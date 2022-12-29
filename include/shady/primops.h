#ifndef SHADY_IR_H
#error "do not include this file by itself, include shady/ir.h instead"
#endif

// All the primops are defined in this file, using the following syntax:
// PRIMOP(has_side_effects, name)

#define PRIMOPS(P)              \
P(0, quote)                     \
P(0, add)                       \
P(0, sub)                       \
P(0, mul)                       \
P(0, div)                       \
P(0, mod)                       \
P(0, neg)                       \
P(0, gt)                        \
P(0, gte)                       \
P(0, lt)                        \
P(0, lte)                       \
P(0, eq)                        \
P(0, neq)                       \
P(0, and)                       \
P(0, or)                        \
P(0, xor)                       \
P(0, not)                       \
P(0, rshift_logical)            \
P(0, rshift_arithm)             \
P(0, lshift)                    \
P(1, assign)                    \
P(1, subscript)                 \
P(1, alloca)                    \
P(1, alloca_logical)            \
P(1, alloca_subgroup)           \
P(0, load)                      \
P(1, store)                     \
P(0, lea)                       \
P(0, select)                    \
P(0, convert)                   \
P(0, reinterpret)               \
P(0, make)                      \
P(0, extract)                   \
P(0, extract_dynamic)           \
P(1, push_stack)                \
P(1, pop_stack)                 \
P(1, push_stack_uniform)        \
P(1, pop_stack_uniform)         \
P(0, get_stack_pointer)         \
P(0, get_stack_pointer_uniform) \
P(0, get_stack_base)            \
P(0, get_stack_base_uniform)    \
P(1, set_stack_pointer)         \
P(1, set_stack_pointer_uniform) \
P(1, create_joint_point)        \
P(0, subgroup_elect_first)      \
P(0, subgroup_broadcast_first)  \
P(0, subgroup_reduce_sum)       \
P(0, subgroup_active_mask)      \
P(0, subgroup_ballot)           \
P(0, subgroup_local_id)         \
P(0, empty_mask)                \
P(0, mask_is_thread_active)     \
P(0, subgroup_id)               \
P(0, workgroup_local_id)        \
P(0, workgroup_id)              \
P(0, global_id)                 \
P(0, workgroup_size)            \
P(0, workgroup_num)             \
P(1, debug_printf)              \

typedef enum Op_ {
#define DECLARE_PRIMOP_ENUM(has_side_effects, name) name##_op,
PRIMOPS(DECLARE_PRIMOP_ENUM)
#undef DECLARE_PRIMOP_ENUM
    PRIMOPS_COUNT
} Op;

extern const char* primop_names[];
bool has_primop_got_side_effects(Op op);
