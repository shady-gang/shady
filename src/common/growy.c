#include "growy.h"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static size_t init_size = 4096;

struct Growy_ {
    char* buffer;
    size_t used, size;
};

Growy* shd_new_growy() {
    Growy* g = calloc(1, sizeof(Growy));
    *g = (Growy) {
        .buffer = calloc(1, init_size),
        .size = init_size,
        .used = 0
    };
    return g;
}

void shd_growy_append_bytes(Growy* g, size_t s, const char* bytes) {
    size_t old_used = g->used;
    g->used += s;
    while (g->used >= g->size) {
        g->size *= 2;
        g->buffer = realloc(g->buffer, g->size);
    }
    memcpy(g->buffer + old_used, bytes, s);
}

void shd_growy_append_string(Growy* g, const char* str) {
    size_t len = strlen(str);
    shd_growy_append_bytes(g, len, str);
}

void shd_format_string_internal(const char* str, va_list args, void* uptr, void callback(void*, size_t, char*));

void shd_growy_append_formatted(Growy* g, const char* str, ...) {
    va_list args;
    va_start(args, str);
    shd_format_string_internal(str, args, g, (void (*)(void*, size_t, char*)) shd_growy_append_bytes);
    va_end(args);
}

void shd_destroy_growy(Growy* g) {
    free(g->buffer);
    free(g);
}

char* shd_growy_deconstruct(Growy* g) {
    char* buf = g->buffer;
    free(g);
    return buf;
}

size_t shd_growy_size(const Growy* g) { return g->used; }
char* shd_growy_data(const Growy* g) { return g->buffer; }
