#ifndef SHADY_PASSES_H

#include "ir.h"

const struct Program* bind_program(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Program* src_program);
const struct Program* type_program(struct IrArena* src_arena, struct IrArena* dst_arena, const struct Program* src_program);

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
