#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
