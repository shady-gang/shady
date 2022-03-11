#ifndef SHADY_IMPLEM_H
#define SHADY_IMPLEM_H

#include "ir.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;

    IrConfig config;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
} IrArena_;

void* arena_alloc(IrArena* arena, size_t size);

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
