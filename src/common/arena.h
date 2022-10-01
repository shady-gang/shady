#ifndef SHADY_ARENA
#define SHADY_ARENA

#include <stddef.h>

typedef struct Arena_ Arena;

Arena* new_arena();
void destroy_arena(Arena* arena);
void* arena_alloc(Arena* arena, size_t size);

#endif
