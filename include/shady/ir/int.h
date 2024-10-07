#ifndef SHADY_IR_INT_H
#define SHADY_IR_INT_H

#include "shady/ir/base.h"

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

static inline int int_size_in_bytes(IntSizes s) {
    switch (s) {
        case IntTy8: return 1;
        case IntTy16: return 2;
        case IntTy32: return 4;
        case IntTy64: return 8;
    }
}

const Type* shd_int_type_helper(IrArena* a, bool s, IntSizes w);

const Type* shd_int8_type(IrArena* arena);
const Type* shd_int16_type(IrArena* arena);
const Type* shd_int32_type(IrArena* arena);
const Type* shd_int64_type(IrArena* arena);

const Type* shd_uint8_type(IrArena* arena);
const Type* shd_uint16_type(IrArena* arena);
const Type* shd_uint32_type(IrArena* arena);
const Type* shd_uint64_type(IrArena* arena);

const Node* shd_int8_literal(IrArena* arena, int8_t i);
const Node* shd_int16_literal(IrArena* arena, int16_t i);
const Node* shd_int32_literal(IrArena* arena, int32_t i);
const Node* shd_int64_literal(IrArena* arena, int64_t i);

const Node* shd_uint8_literal(IrArena* arena, uint8_t u);
const Node* shd_uint16_literal(IrArena* arena, uint16_t u);
const Node* shd_uint32_literal(IrArena* arena, uint32_t u);
const Node* shd_uint64_literal(IrArena* arena, uint64_t u);

#endif
