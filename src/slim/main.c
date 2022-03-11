#include <stdio.h>
#include <stdlib.h>

#include "ir.h"
#include "token.h"

#include "../passes/passes.h"

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

    IrArena* arena = new_arena((IrConfig) {
        .check_types = false
    });

    if (argc <= 1)
        printf("Usage: slim source.slim\n");
    else {
        const Node* program;

        char* filename = argv[1];
        printf("compiling %s\n", filename);

        char* contents = read_file(filename);
        if ((void*)contents == NULL) {
            printf("file does not exist\n");
            return -1;
        } else {
            printf("Parsing: \n%s\n", contents);
            program = parse(contents, arena);
        }

        free(contents);

        printf("Parsed program successfully: \n");
        print_node(program);

        program = bind_program(arena, arena, program);
        printf("Bound program successfully: \n");
        print_node(program);

        IrArena* typed_arena = new_arena((IrConfig) {
            .check_types = true
        });
        program = type_program(arena, typed_arena, program);
        destroy_arena(arena);
        arena = typed_arena;
        printf("Typed program successfully: \n");
        print_node(program);

        FILE *output = fopen("out.spv", "wb");
        emit(program, output);
        fclose(output);
    }

    destroy_arena(arena);
    return 0;
}
