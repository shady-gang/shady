#include <stdlib.h>
#include <stdio.h>

char* read_file(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        return NULL;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

    char *string = malloc(fsize + 1);
    fread(string, fsize, 1, f);
    fclose(f);

    string[fsize] = 0;
    return string;
}
