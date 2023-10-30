#include "util.h"
#include "arena.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define ESCAPE_SEQS(X) \
X('\\', '\\') \
X('\'', '\'') \
X('\"', '\"') \
X( 'n', '\n') \
X( 't', '\t') \
X( 'b', '\b') \
X( 'r', '\r') \
X( 'f', '\f') \
X( 'a', '\a') \
X( 'v', '\v') \

size_t apply_escape_codes(const char* src, size_t size, char* dst) {
    char p, c = '\0';
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        p = c;
        c = src[i];

#define ESCAPE_CASE(m, s) if (p == '\\' && c == m) { \
        dst[j - 1] = s; \
        continue; \
    } \

        ESCAPE_SEQS(ESCAPE_CASE)

        dst[j++] = c;
    }
    return j;
}

static long get_file_size(FILE* f) {
    if (fseek(f, 0, SEEK_END) != 0)
        return -1;

    long fsize = ftell(f);

    if (fsize == -1)
        return -1;

    if (fseek(f, 0, SEEK_SET) != 0)  /* same as rewind(f); */
        return -1;

    return fsize;
}

bool read_file(const char* filename, size_t* size, char** output) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        return false;

    long fsize = get_file_size(f);
    if (fsize < 0)
        goto err_post_open;

    // pad an extra zero at the end so this can be safely treated like a string
    unsigned char* string = malloc(fsize + 1);
    if (!string)
        goto err_post_open;

    if (fread(string, fsize, 1, f) != 1)
        goto err_post_alloc;

    fclose(f);

    string[fsize] = 0;
    if (output)
        *output = string;
    if (size)
        *size = fsize;
    return true;

err_post_alloc:
    free(string);
err_post_open:
    fclose(f);
    return false;
}

bool write_file(const char* filename, size_t size, const char* data) {
    FILE* f = fopen(filename, "wb");
    if (f == NULL)
        return false;

    if (fwrite(data, size, 1, f) != 1)
        goto err_post_open;

    fclose(f);

    return true;

err_post_open:
    fclose(f);
    return false;
}

enum {
    ThreadLocalStaticBufferSize = 256
};

static char static_buffer[ThreadLocalStaticBufferSize];

char* format_string(Arena* arena, const char* str, ...) {
    size_t buffer_size = ThreadLocalStaticBufferSize;
    int len;
    char* tmp;
    while (true) {
        if (buffer_size == ThreadLocalStaticBufferSize) {
            tmp = static_buffer;
        } else {
            tmp = malloc(buffer_size);
        }
        va_list args;
        va_start(args, str);
        len = vsnprintf(tmp, buffer_size, str, args);
        va_end(args);
        if (len < 0 || len >= (int) buffer_size - 1) {
            buffer_size *= 2;
            if (tmp != static_buffer)
                free(tmp);
            continue;
        } else {
            // const char* interned = string_impl(arena, len, tmp);
            char* interned = (char*) arena_alloc(arena, len + 1);
            strncpy(interned, tmp, len);
            interned[len] = '\0';
            if (tmp != static_buffer)
                free(tmp);
            return interned;
        }
    }
}

bool string_starts_with(const char* string, const char* prefix) {
    size_t len = strlen(string);
    size_t slen = strlen(prefix);
    if (len < slen)
        return false;
    return memcmp(string, prefix, slen) == 0;
}

bool string_ends_with(const char* string, const char* suffix) {
    size_t len = strlen(string);
    size_t slen = strlen(suffix);
    if (len < slen)
        return false;
    for (size_t i = 0; i < slen; i++) {
        if (string[len - 1 - i] != suffix[slen - 1 - i])
            return false;
    }
    return true;
}

void error_die() {
    abort();
}
