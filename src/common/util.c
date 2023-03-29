#include "util.h"

#include <stdlib.h>
#include <stdio.h>

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

bool read_file(const char* filename, size_t* size, unsigned char** output) {
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

bool write_file(const char* filename, size_t size, const unsigned char* data) {
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

void error_die() {
    abort();
}
