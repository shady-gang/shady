#ifndef SHADY_IMPLEM_H
#define SHADY_IMPLEM_H

#include "ir.h"

#include "stdlib.h"
#include "stdio.h"

void* arena_alloc(struct IrArena* arena, size_t size);

#define NODEDEF(struct_name, short_name) const struct Type* infer_##short_name(struct IrArena*, struct struct_name);
NODES()
#undef NODEDEF

#ifdef _MSC_VER
#define SHADY_UNREACHABLE __assume(0)
#else
#define SHADY_UNREACHABLE __builtin_unreachable()
#endif

#define error(...) {             \
  fprintf (stderr, __VA_ARGS__); \
  exit(-1);                      \
  SHADY_UNREACHABLE;             \
}

#endif
