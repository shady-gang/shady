#ifndef _SHADY_META_H
#define _SHADY_META_H

#include <stdint.h>

#define SHADY_META_IDS_BEGIN_AT 0x1000

typedef enum {
    SHADY_META_INVALID,
    SHADY_META_DEFINE_LITERAL_I32,
    SHADY_META_DEFINE_LITERAL_STRING,
    SHADY_META_DEFINE_BUILTIN_TYPE,
    SHADY_META_DEFINE_PARAM_REF,
    SHADY_META_DEFINE_EXT_OP,
} shady_meta_instruction;

typedef uint32_t shady_meta_id;

typedef struct {
    shady_meta_instruction meta;
    shady_meta_id defined_id;
    uint32_t literal;
} shady_meta_literal_i32;

#define __shady_define_literal_i32(id, name, value) \
shady_meta_literal_i32 __shady_meta_op_##name = { SHADY_META_DEFINE_LITERAL_I32, id, value }; \
static const uint32_t __shady_result_id_##name = id;

typedef struct {
    shady_meta_instruction meta;
    shady_meta_id defined_id;
    const char* literal;
} shady_meta_literal_string;

#define __shady_define_literal_string(id, name, value) \
shady_meta_literal_string __shady_meta_op_##name = { SHADY_META_DEFINE_LITERAL_STRING, id, value }; \
static const uint32_t __shady_result_id_##name = id;

#define __shady_define_builtin_type(id, name, T) \
typedef struct { \
    shady_meta_instruction meta; \
    shady_meta_id defined_id; \
    T dummy; \
} shady_meta_builtin_type_##name; \
shady_meta_builtin_type_##name __shady_meta_op_##name = { SHADY_META_DEFINE_BUILTIN_TYPE, id }; \
static const uint32_t __shady_result_id_##name = id;

typedef struct {
    shady_meta_instruction meta;
    shady_meta_id defined_id;
    unsigned param_idx;
} shady_meta_param_ref;

#define __shady_define_param_ref(id, name, idx) \
shady_meta_param_ref __shady_meta_op_##name = { SHADY_META_DEFINE_PARAM_REF, id, idx }; \
static const uint32_t __shady_result_id_##name = id;

typedef struct {
    shady_meta_instruction meta;
    shady_meta_id defined_id;
    uint32_t op_code;
    /* only used in the parser */
    uint32_t num_operands;
    shady_meta_id* operands;
} shady_meta_ext_op;

#define __shady_define_ext_op(id, name, op, ...) \
shady_meta_ext_op __shady_meta_op_##name = { SHADY_META_DEFINE_EXT_OP, id, op, 0, (uint32_t[]) { __VA_ARGS__ } }; \
static const uint32_t __shady_result_id_##name = id;

#define __shady_define_ext_type(id, name, op, ...) \
typedef __attribute__((address_space(id))) struct __shady_id_as_addrspace_##name* name; \
__shady_define_ext_op(id, name, op, __VA_ARGS__)

#define __shady_define_ext_inst(id, name, op, ...) \
__shady_define_ext_op(id, name, op, __VA_ARGS__)

#define __shady_declare_literal_i32(name, value)    __shady_define_literal_i32(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, value)
#define __shady_declare_literal_string(name, value) __shady_define_literal_string(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, value)
#define __shady_declare_builtin_type(name, T)       __shady_define_builtin_type(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, T)
#define __shady_declare_param_ref(name, i)          __shady_define_param_ref(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, i)
#define __shady_declare_ext_type(name, op, ...)     __shady_define_ext_type(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, op, __VA_ARGS__)
#define __shady_declare_ext_inst(name, op, ...)     __shady_define_ext_inst(SHADY_META_IDS_BEGIN_AT + __COUNTER__, name, op, __VA_ARGS__)

// you cannot refer to a shady def directly, you must reference it using this macro
#define __shady_ref(name) __shady_result_id_##name
#define __shady_bind_ext_inst(name) __asm__("shady::meta_ext_inst::"#name);

#endif
