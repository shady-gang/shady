#ifndef SHADY_PASSES_H

#include "ir.h"

struct Program bind_variables(struct IrArena* src_arena, struct IrArena* dst_arena, struct Program* src_program);
struct Program infer_types(struct IrArena* src_arena, struct IrArena* dst_arena, struct Program* src_program);

#define SHADY_PASSES_H

#endif //SHADY_PASSES_H
