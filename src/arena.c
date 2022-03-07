#include "ir.h"
#include "implem.h"

#include "stdlib.h"
#include "string.h"
#include "assert.h"

#define alloc_size 1024 * 1024

struct IrArena* new_arena() {
    struct IrArena* arena = malloc(sizeof(struct IrArena));
    *arena = (struct IrArena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256 * sizeof(size_t)),
        .available = 0,
        .type_table = new_type_table()
    };
    for (int i = 0; i < arena->maxblocks; i++)
        arena->blocks[i] = NULL;
    return arena;
}

void destroy_arena(struct IrArena* arena) {
    destroy_type_table(arena->type_table);
    for (int i = 0; i < arena->nblocks; i++) {
        free(arena->blocks[i]);
    }
    free(arena->blocks);
    free(arena);
}

void* arena_alloc(struct IrArena* arena, size_t size) {
    if (size == 0)
        return NULL;
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
    memset(allocated, 0, size);
    arena->available -= size;
    return allocated;
}

struct IrArena* rebuild_arena(struct IrArena* arena) {
    SHADY_NOT_IMPLEM
}

struct Nodes reserve_nodes(struct IrArena* arena, size_t count) {
    struct Nodes nodes = {
        .count = count,
        .nodes = arena_alloc(arena, count * sizeof(size_t))
    };
    return nodes;
}

struct Variables reserve_variables(struct IrArena* arena, size_t count) {
    struct Variables variables = {
        .count = count,
        .variables = arena_alloc(arena, count * sizeof(size_t))
    };
    return variables;
}

struct Types reserve_types(struct IrArena* arena, size_t count)  {
    struct Types types = {
        .count = count,
        .types = arena_alloc(arena, count * sizeof(size_t))
    };
    return types;
}

struct Nodes nodes(struct IrArena* arena, size_t count, const struct Node* in_nodes[]) {
    struct Nodes nodes = reserve_nodes(arena, count);
    for (size_t i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];
    return nodes;
}

struct Variables variables(struct IrArena* arena, size_t count, const struct Variable* in_vars[]) {
    struct Variables variables = reserve_variables(arena, count);
    for (size_t i = 0; i < count; i++)
        variables.variables[i] = in_vars[i];
    return variables;
}

struct Types types(struct IrArena* arena, size_t count, const struct Type* in_types[])  {
    struct Types types = reserve_types(arena, count);
    for (size_t i = 0; i < count; i++)
        types.types[i] = in_types[i];
    return types;
}

const char* string(struct IrArena* arena, size_t size, const char* str) {
    char* new_str = (char*) arena_alloc(arena, size + 1);
    strncpy(new_str, str, size);
    new_str[size] = 0;
    assert(strlen(new_str) == size);
    return new_str;
}
