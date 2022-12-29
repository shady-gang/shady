#include "growy.h"

#include <stdlib.h>
#include <string.h>

static size_t init_size = 4096;

struct Growy_ {
    char* buffer;
    size_t used, size;
};

Growy* new_growy() {
    Growy* g = calloc(1, sizeof(Growy));
    *g = (Growy) {
        .buffer = calloc(1, init_size),
        .size = init_size,
        .used = 0
    };
    return g;
}

void growy_append_bytes(Growy* g, size_t s, const char* bytes) {
    size_t old_used = g->used;
    g->used += s;
    while (g->used >= g->size) {
        g->size *= 2;
        g->buffer = realloc(g->buffer, g->size);
    }
    memcpy(g->buffer + old_used, bytes, s);
}

void destroy_growy(Growy* g) {
    free(g->buffer);
    free(g);
}

char* growy_deconstruct(Growy* g) {
    char* buf = g->buffer;
    free(g);
    return buf;
}

size_t growy_size(const Growy* g) { return g->used; }
char* growy_data(const Growy* g) { return g->buffer; }
