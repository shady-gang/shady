#include <stdio.h>
#include <stdlib.h>

#include "ir.h"
#include "token.h"

struct Program parse(char* contents, struct IrArena* arena);

char* read_file(char* filename) {
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

int main(int argc, char** argv) {
    init_tokenizer_constants();

    struct IrArena* arena = new_arena();

    if (argc <= 1)
        printf("Usage: slim source.slim\n");
    else {
        char* filename = argv[1];
        printf("compiling %s\n", filename);

        char* contents = read_file(filename);
        if ((void*)contents == NULL) {
            printf("file does not exist\n");
            return -1;
        } else {
            printf("Parsing: \n%s\n", contents);
            struct Program program = parse(contents, arena);
            print_program(&program);
        }

        free(contents);
    }

    destroy_arena(arena);
    return 0;
}
