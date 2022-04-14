#include "ir.h"
#include "token.h"

#include "../implem.h"
#include "../passes/passes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MoreThanOneFilename
};

const char* input_filename = NULL;
const char* output_filename = "out.spv";

static char* read_file(char* filename) {
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

static void process_arguments(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log-level") == 0) {
            i++;
            if (i == argc)
                goto incorrect_log_level;
            if (strcmp(argv[i], "debug") == 0)
                set_log_level(DEBUG);
            else if (strcmp(argv[i], "info") == 0)
                set_log_level(INFO);
            else if (strcmp(argv[i], "warn") == 0)
                set_log_level(WARN);
            else if (strcmp(argv[i], "error") == 0)
                set_log_level(ERROR);
            else {
                incorrect_log_level:
                error_print("--log-level argument takes one of: ");
                error_print("debug, info, warn,  error");
                error_print("\n");
                exit(IncorrectLogLevel);
            }
        } else if (strcmp(argv[i], "--output") == 0) {
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            output_filename = argv[i];
        } else {
            // assume it is the filename
            if (input_filename) {
                error_print("We only supporting compiling from one file at the moment");
                exit(MoreThanOneFilename);
            }
            input_filename = argv[i];
        }
    }

    if (input_filename == NULL) {
        error_print("Usage: slim source.slim\n");
        exit(MissingInputArg);
    }
}

int main(int argc, char** argv) {
    init_tokenizer_constants();

    IrArena* arena = new_arena((IrConfig) {
        .check_types = false
    });

    process_arguments(argc, argv);

    info_print("compiling %s\n", input_filename);
    const Node* program;

    char* contents = read_file(input_filename);
    if ((void*)contents == NULL) {
        error_print("file does not exist\n");
        exit(InputFileDoesNotExist);
    } else {
        debug_print("Parsing: \n%s\n", contents);
        program = parse(contents, arena);
    }

    free(contents);

    debug_print("Parsed program successfully: \n");
    print_node(program);

    program = bind_program(arena, arena, program);
    debug_print("Bound program successfully: \n");
    debug_node(program);

    IrArena* typed_arena = new_arena((IrConfig) {
        .check_types = true
    });
    program = type_program(arena, typed_arena, program);
    destroy_arena(arena);
    arena = typed_arena;
    debug_print("Typed program successfully: \n");
    debug_node(program);

    program = instr2bb(arena, arena, program);
    debug_print("instr2bb pass: \n");
    debug_node(program);

    debug_print("Emitting final result ... \n");
    FILE *output = fopen(output_filename, "wb");
    emit(arena, program, output);
    fclose(output);
    debug_print("Done\n");

    destroy_arena(arena);
    return NoError;
}
