#ifndef SHADY_ARENA
#define SHADY_ARENA

#include <stddef.h>

typedef struct Arena_ Arena;

Arena* shd_new_arena(void);
void shd_destroy_arena(Arena* arena);
void* shd_arena_alloc(Arena* arena, size_t size);

#endif
