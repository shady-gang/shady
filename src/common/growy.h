#ifndef SHADY_GROWY_H
#define SHADY_GROWY_H

#include <stddef.h>

/// Growy buffer, a buffer that can grow.
/// Addresses not guaranteed to be stable.
typedef struct Growy_ Growy;

Growy* shd_new_growy(void);
void shd_growy_append_bytes(Growy*, size_t, const char*);
void shd_growy_append_string(Growy* g, const char* str);
void shd_growy_append_formatted(Growy* g, const char* str, ...);
#define shd_growy_append_string_literal(a, v) shd_growy_append_bytes(a, sizeof(v) - 1, (char*) &v)
#define shd_growy_append_object(a, v) shd_growy_append_bytes(a, sizeof(v), (char*) &v)
size_t shd_growy_size(const Growy* g);
char* shd_growy_data(const Growy* g);
void shd_destroy_growy(Growy*g);
// Like destroy, but we scavenge the internal allocation for later use.
char* shd_growy_deconstruct(Growy* g);

#endif
