#include "util.h"

#include <stdlib.h>
#include <stdio.h>

bool read_file(const char* filename, size_t* size, unsigned char** output) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        return false;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

    // pad an extra zero at the end so this can be safely treated like a string
    unsigned char* string = malloc(fsize + 1);
    fread(string, fsize, 1, f);
    fclose(f);

    string[fsize] = 0;
    if (output)
        *output = string;
    if (size)
        *size = fsize;
    return true;
}

void error_die() {
    abort();
}
