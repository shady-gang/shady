#include "util.h"
#include "printer.h"
#include "arena.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

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

size_t shd_apply_escape_codes(const char* src, size_t size, char* dst) {
    char prev, c = '\0';
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        prev = c;
        c = src[i];

#define ESCAPE_CASE(m, s) if (prev == '\\' && c == m) { \
        dst[j - 1] = s; \
        continue; \
    } \

        ESCAPE_SEQS(ESCAPE_CASE)
#undef ESCAPE_CASE

        dst[j++] = c;
    }
    return j;
}

size_t shd_unapply_escape_codes(const char* src, size_t size, char* dst) {
    char c = '\0';
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        c = src[i];

#define ESCAPE_CASE(m, s) if (c == s) { \
        dst[j++] = '\\'; \
        dst[j++] = m; \
        continue; \
    } \

        ESCAPE_SEQS(ESCAPE_CASE)
#undef ESCAPE_CASE

        dst[j++] = c;
    }
    return j;
}

void shd_printer_escape(Printer* p, const char* src) {
    size_t size = strlen(src);
    for (size_t i = 0; i < size; i++) {
        char c = src[i];
        char next = i + 1 < size ? src[i + 1] : '\0';

#define ESCAPE_CASE(m, s) if (c == '\\' && next == m) { \
        char code = s; \
        shd_print(p, "%c", code); \
        i++; \
        continue; \
    } \

        ESCAPE_SEQS(ESCAPE_CASE)
#undef ESCAPE_CASE
        shd_print(p, "%c", c);
    }
}

void shd_printer_unescape(Printer* p, const char* src) {
    size_t size = strlen(src);
    for (size_t i = 0; i < size; i++) {
        char c = src[i];

#define ESCAPE_CASE(m, s) if (c == s) { \
        shd_print(p, "\\%c", m); \
        continue; \
    } \

        ESCAPE_SEQS(ESCAPE_CASE)
#undef ESCAPE_CASE
        shd_print(p, "%c", c);
    }
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

bool shd_read_file(const char* filename, size_t* size, char** output) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        return false;

    long fsize = get_file_size(f);
    if (fsize < 0)
        goto err_post_open;

    // pad an extra zero at the end so this can be safely treated like a string
    char* string = malloc(fsize + 1);
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

bool shd_write_file(const char* filename, size_t size, const char* data) {
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

void shd_format_string_internal(const char* str, va_list args, void* uptr, void callback(void*, size_t, char*)) {
    size_t buffer_size = ThreadLocalStaticBufferSize;
    int len;
    char* tmp;
    while (true) {
        if (buffer_size == ThreadLocalStaticBufferSize) {
            tmp = static_buffer;
        } else {
            tmp = malloc(buffer_size);
        }
        va_list args_copy;
        va_copy(args_copy, args);
        len = vsnprintf(tmp, buffer_size, str, args_copy);
        if (len < 0 || len >= (int) buffer_size - 1) {
            buffer_size *= 2;
            if (tmp != static_buffer)
                free(tmp);
            continue;
        } else {
            callback(uptr, len, tmp);
            if (tmp != static_buffer)
                free(tmp);
            return;
        }
    }
}

typedef struct { Arena* a; char** result; } InternInArenaPayload;

static void intern_in_arena(InternInArenaPayload* uptr, size_t len, char* tmp) {
    char* interned = (char*) shd_arena_alloc(uptr->a, len + 1);
    strncpy(interned, tmp, len);
    interned[len] = '\0';
    *uptr->result = interned;
}

char* shd_format_string_arena(Arena* arena, const char* str, ...) {
    char* result = NULL;
    InternInArenaPayload p = { .a = arena, .result = &result };
    va_list args;
    va_start(args, str);
    shd_format_string_internal(str, args, &p, (void (*)(void*, size_t, char*)) intern_in_arena);
    va_end(args);
    return result;
}

typedef struct { char** result; } PutNewPayload;

static void put_in_new(PutNewPayload* uptr, size_t len, char* tmp) {
    char* allocated = (char*) malloc(len + 1);
    strncpy(allocated, tmp, len);
    allocated[len] = '\0';
    *uptr->result = allocated;
}

char* shd_format_string_new(const char* str, ...) {
    char* result = NULL;
    PutNewPayload p = { .result = &result };
    va_list args;
    va_start(args, str);
    shd_format_string_internal(str, args, &p, (void (*)(void*, size_t, char*)) put_in_new);
    va_end(args);
    return result;
}

bool shd_string_starts_with(const char* string, const char* prefix) {
    size_t len = strlen(string);
    size_t slen = strlen(prefix);
    if (len < slen)
        return false;
    return memcmp(string, prefix, slen) == 0;
}

bool shd_string_ends_with(const char* string, const char* suffix) {
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

char* shd_strip_path(const char* path) {
    char separator = strchr(path, '\\') == NULL ? '/' : '\\';
    char* end = strrchr(path, separator);
    if (!end) {
        fprintf(stderr, "path: %s\n", path);
        char* new = calloc(3, sizeof(char));
        new[0] = '.';
        new[1] = '/';
        return new;
    }
    char* new = calloc((end - path) + 1, sizeof(char));
    size_t i = 0;
    for (const char* c = path; c < end; c++) {
        new[i++] = *c;
    }
    return new;
}


static bool safe_substr(const char* str, size_t start, size_t end, const char* needle) {
    size_t needle_len = strlen(needle);
    return end - start >= needle_len && memcmp(&str[start], needle, needle_len) == 0;
}

void shd_configure_bool_flag_in_list(const char* str, const char* flag_name, bool* flag_value) {
    if (!str)
        return;
    size_t len = strlen(str);
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || str[i] == ',') {
            size_t sublen = i - start;
            if (safe_substr(str, start, i, flag_name)) {
                if (strlen(flag_name) + 1 < sublen) {
                    // eat the '='
                    if (safe_substr(str, start + strlen(flag_name) + 1, i, "1"))
                        *flag_value = true;
                    if (safe_substr(str, start + strlen(flag_name) + 1, i, "0"))
                        *flag_value = false;
                } else {
                    *flag_value = true;
                }
            }
            start = i + 1;
        }
    }
}

void shd_configure_int_flag_in_list(const char* str, const char* flag_name, int* flag_value) {
    if (!str)
        return;
    size_t len = strlen(str);
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || str[i] == ',') {
            size_t sublen = i - start;
            if (safe_substr(str, start, i, flag_name)) {
                size_t flag_len = strlen(flag_name);
                if (flag_len + 1 < sublen) {
                    // eat the '='
                    *flag_value = strtol(&str[1 + flag_len], NULL, 10);
                }
            }
            start = i + 1;
        }
    }
}

void shd_error_die(void) {
    abort();
}
