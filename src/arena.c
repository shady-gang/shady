#include "ir.h"
#include "implem.h"

#include "stdlib.h"
#include "string.h"
#include "assert.h"

#define alloc_size 1024 * 1024

struct IrArena* new_arena(struct IrConfig config) {
    struct IrArena* arena = malloc(sizeof(struct IrArena));
    *arena = (struct IrArena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256 * sizeof(size_t)),
        .available = 0,
        .config = config,
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

struct Nodes reserve_nodes(struct IrArena* arena, size_t count) {
    struct Nodes nodes = {
        .count = count,
        .nodes = arena_alloc(arena, count * sizeof(size_t))
    };
    return nodes;
}

struct Types reserve_types(struct IrArena* arena, size_t count)  {
    struct Types types = {
        .count = count,
        .types = arena_alloc(arena, count * sizeof(size_t))
    };
    return types;
}

struct Strings reserve_strings(struct IrArena* arena, size_t count)  {
    struct Strings strings = {
        .count = count,
        .strings = arena_alloc(arena, count * sizeof(size_t))
    };
    return strings;
}

struct Nodes nodes(struct IrArena* arena, size_t count, const struct Node* in_nodes[]) {
    struct Nodes nodes = reserve_nodes(arena, count);
    for (size_t i = 0; i < count; i++)
        nodes.nodes[i] = in_nodes[i];
    return nodes;
}

struct Types types(struct IrArena* arena, size_t count, const struct Type* in_types[])  {
    struct Types types = reserve_types(arena, count);
    for (size_t i = 0; i < count; i++)
        types.types[i] = in_types[i];
    return types;
}

struct Strings strings(struct IrArena* arena, size_t count, const char* in_strs[])  {
    struct Strings strings = reserve_strings(arena, count);
    for (size_t i = 0; i < count; i++)
        strings.strings[i] = in_strs[i];
    return strings;
}

const char* string_sized(struct IrArena* arena, size_t size, const char* str) {
    char* new_str = (char*) arena_alloc(arena, size + 1);
    strncpy(new_str, str, size);
    new_str[size] = '\0';
    assert(strlen(new_str) == size);
    return new_str;
}

const char* string(struct IrArena* arena, const char* str) {
    return string_sized(arena, strlen(str), str);
}

