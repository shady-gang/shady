#ifndef SHADY_IMPLEM_H
#define SHADY_IMPLEM_H

#include "ir.h"

#include "stdlib.h"
#include "stdio.h"

struct IrArena {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;

    struct IrConfig config;

    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* types_set;
    struct Dict* strings_set;

    struct TypeTable* type_table;
};

void* arena_alloc(struct IrArena* arena, size_t size);

#define NODEDEF(struct_name, short_name) const struct Type* infer_##short_name(struct IrArena*, struct struct_name);
NODES()
#undef NODEDEF

#ifdef _MSC_VER
#define SHADY_UNREACHABLE __assume(0)
#else
#define SHADY_UNREACHABLE __builtin_unreachable()
#endif

#define SHADY_NOT_IMPLEM {    \
  error("not implemented\n"); \
  SHADY_UNREACHABLE;          \
}

#define error(...) {             \
  fprintf (stderr, __VA_ARGS__); \
  exit(-1);                      \
}

#endif
