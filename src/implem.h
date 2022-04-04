#ifndef SHADY_IMPLEM_H
#define SHADY_IMPLEM_H

#include "ir.h"

#include "local_array.h"

#include "stdlib.h"
#include "stdio.h"

typedef struct IrArena_ {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;

    IrConfig config;

    VarId next_free_id;

    struct Dict* node_set;
    struct Dict* string_set;

    struct Dict* nodes_set;
    struct Dict* strings_set;
} IrArena_;

void* arena_alloc(IrArena* arena, size_t size);

VarId fresh_id(IrArena*);


#ifdef NDEBUG
#ifdef _MSC_VER
#define SHADY_UNREACHABLE __assume(0)
#else
#define SHADY_UNREACHABLE __builtin_unreachable()
#endif
#else
#define SHADY_UNREACHABLE exit(69)
#endif

#define SHADY_NOT_IMPLEM {    \
  error("not implemented\n"); \
  SHADY_UNREACHABLE;          \
}

#define error(...) {             \
  fprintf (stderr, __VA_ARGS__); \
  error_impl();                  \
  SHADY_UNREACHABLE;             \
}

static void error_impl() {
    exit(-1);
}

#endif
