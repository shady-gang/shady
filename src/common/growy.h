#ifndef SHADY_GROWY_H
#define SHADY_GROWY_H

#include <stddef.h>

/// Growy buffer, a buffer that can grow.
/// Addresses not guaranteed to be stable.
typedef struct Growy_ Growy;

Growy* new_growy();
void growy_append_bytes(Growy*, size_t, const char*);
#define growy_append(a, v) growy_append_bytes(a, sizeof(v), (char*) &v)
size_t growy_size(const Growy*);
char* growy_data(const Growy*);
void growy_destroy(Growy*);
// Like destroy, but we scavenge the internal allocation for later use.
char* growy_deconstruct(Growy*);

#endif
