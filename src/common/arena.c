#include "arena.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define alloc_size 1024 * 1024

typedef struct Arena_ {
    int nblocks;
    int maxblocks;
    void** blocks;
    size_t available;
} Arena;

inline static size_t round_up(size_t a, size_t b) {
    size_t divided = (a + b - 1) / b;
    return divided * b;
}

Arena* shd_new_arena() {
    Arena* arena = malloc(sizeof(Arena));
    *arena = (Arena) {
        .nblocks = 0,
        .maxblocks = 256,
        .blocks = malloc(256 * sizeof(void*)),
        .available = 0,
    };
    for (int i = 0; i < arena->maxblocks; i++)
        arena->blocks[i] = NULL;
    return arena;
}

void shd_destroy_arena(Arena* arena) {
    for (int i = 0; i < arena->nblocks; i++) {
        free(arena->blocks[i]);
    }
    free(arena->blocks);
    free(arena);
}

static void* new_block(Arena* arena, size_t size) {
    assert(arena->nblocks <= arena->maxblocks);
    // we need more storage for the block pointers themselves !
    if (arena->nblocks == arena->maxblocks) {
        arena->maxblocks *= 2;
        arena->blocks = realloc(arena->blocks, arena->maxblocks * sizeof(void*));
    }

    void* allocated = malloc(size);
    assert(allocated);
    arena->blocks[arena->nblocks++] = allocated;
    return allocated;
}

void* shd_arena_alloc(Arena* arena, size_t size) {
    size = round_up(size, (size_t) sizeof(max_align_t));
    if (size == 0)
        return NULL;
    if (size > alloc_size) {
        void* allocated = new_block(arena, size);
        memset(allocated, 0, size);
        // swap the last two blocks
        if (arena->nblocks >= 2) {
            void* swap = arena->blocks[arena->nblocks - 1];
            arena->blocks[arena->nblocks - 1] = arena->blocks[arena->nblocks - 2];
            arena->blocks[arena->nblocks - 2] = swap;
        }
        return allocated;
    }

    // arena is full
    if (size > arena->available) {
        new_block(arena, alloc_size);
        arena->available = alloc_size;
    }

    assert(size <= arena->available);

    size_t in_block = alloc_size - arena->available;
    void* allocated = (void*) ((size_t) arena->blocks[arena->nblocks - 1] + in_block);
    memset(allocated, 0, size);
    arena->available -= size;
    return allocated;
}
