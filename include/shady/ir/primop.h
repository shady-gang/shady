#ifndef SHADY_IR_PRIMOP_H
#define SHADY_IR_PRIMOP_H

#include "shady/ir/grammar.h"

OpClass shd_get_primop_class(Op op);

String shd_get_primop_name(Op op);

const Node* shd_op_fma(IrArena* arena, const Node* a, const Node* b, const Node* c);
const Node* shd_op_fabs(IrArena* arena, const Node* a);
const Node* shd_op_floor(IrArena* arena, const Node* a);
const Node* shd_op_ceil(IrArena* arena, const Node* a);
const Node* shd_op_umax(IrArena* arena, const Node* a, const Node* b);
const Node* shd_op_smax(IrArena* arena, const Node* a, const Node* b);
const Node* shd_op_umin(IrArena* arena, const Node* a, const Node* b);
const Node* shd_op_smin(IrArena* arena, const Node* a, const Node* b);

#endif
