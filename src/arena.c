#include "ir.h"
#include "implem.h"

#include "stdlib.h"
#include "string.h"
#include "assert.h"

#define alloc_size 1024 * 1024

struct IrArena {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;
};

struct IrArena* new_arena() {
    struct IrArena* arena = malloc(sizeof(struct IrArena));
    *arena = (struct IrArena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256),
        .available = 0
    };
    for (int i = 0; i < arena->maxblocks; i++)
        arena->blocks[i] = NULL;
    return arena;
}

void destroy_arena(struct IrArena* arena) {
    for (int i = 0; i < arena->nblocks; i++) {
        free(arena->blocks[i]);
    }
    free(arena->blocks);
    free(arena);
}

void* arena_alloc(struct IrArena* arena, size_t size) {
    // arena is full
    if (size > arena->available) {
        assert(arena->nblocks <= arena->maxblocks);
        // we need more storage for the block pointers themselves !
        if (arena->nblocks == arena->maxblocks) {
            arena->maxblocks *= 2;
            arena->blocks = realloc(arena->blocks, arena->maxblocks);
        }

        arena->blocks[arena->nblocks++] = malloc(alloc_size);
        arena->available = alloc_size;
    }

    assert(size <= arena->available);

    size_t in_block = alloc_size - arena->available;
    void* allocated = (void*) ((size_t) arena->blocks[arena->nblocks - 1] + in_block);
    arena->available -= size;
    return allocated;
}

struct IrArena* rebuild_arena(struct IrArena*);

struct Nodes nodes(struct IrArena* arena, int count, struct Node* in_nodes[]) {
    struct Nodes nodes = {
        .count = count,
        .nodes = arena_alloc(arena, count * sizeof(size_t))
    };
    for (int i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];
    return nodes;
}

struct Variables variables(struct IrArena* arena, int count, struct Variable* in_vars[]) {
    struct Variables variables = {
            .count = count,
            .variables = arena_alloc(arena, count * sizeof(size_t))
    };
    for (int i = 0; i < count; i++)
        variables.variables[i] = in_vars[i];
    return variables;
}

struct Types types(struct IrArena* arena, int count, struct Type* in_types[])  {
    struct Types types = {
            .count = count,
            .types = arena_alloc(arena, count * sizeof(size_t))
    };
    for (int i = 0; i < count; i++)
        types.types[i] = in_types[i];
    return types;
}

#define NODEDEF(struct_name, short_name) const struct Node* short_name(struct IrArena* arena, struct struct_name in_node) { \
    struct Node* node = (struct Node*) arena_alloc(arena, sizeof(struct Node));                                             \
    *node = (struct Node) {                                                                                                 \
      .type = infer_##short_name(arena, in_node),                                                                           \
      .tag = struct_name##_TAG,                                                                                             \
      .payload = (union NodesUnion) {                                                                                       \
          .short_name = in_node                                                                                             \
      }                                                                                                                     \
    };                                                                                                                      \
    return node;                                                                                                            \
}

NODES()
#undef NODEDEF

const char* string(struct IrArena* arena, size_t size, char* str) {
    char* new_str = (char*) arena_alloc(arena, size + 1);
    strncpy(new_str, str, size);
    new_str[size] = 0;
    assert(strlen(new_str) == size);
    return new_str;
}
