#include "shady/ir.h"

#include "parser.h"

#include "list.h"

#include "../log.h"
#include "../portability.h"
#include "../passes/passes.h"
#include "../builtin/builtin_code.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

const char* cfg_output = NULL;

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MoreThanOneFilename,
    MissingDumpCfgArg
};

char* read_file(const char* filename);

static void process_arguments(int argc, const char** argv, struct List* input_filenames, const char** output_filename) {
    bool help = false;
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
            *output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            cfg_output = argv[i];
        }  else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            i++;
        } else {
            append_list(const char*, input_filenames, argv[i]);
        }
    }

    if (entries_count_list(input_filenames) == 0 || help) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --output output_filename\n");
        error_print("  --dump-cfg\n");
        exit(help ? 0 : MissingInputArg);
    }
}

static size_t num_builtin_sources_files = 1;
static const char* builtin_source_files[] = { builtin_scheduler_txt };

int main(int argc, const char** argv) {
    platform_specific_terminal_init_extras();

    IrArena* arena = new_arena((ArenaConfig) {
        .check_types = false
    });

    CompilerConfig config = default_compiler_config();

    struct List* input_files = new_list(const char*);
    const char* output_filename = "out.spv";
    process_arguments(argc, argv, input_files, &output_filename);

    size_t num_source_files = num_builtin_sources_files + entries_count_list(input_files);

    LARRAY(const Node*, parsed_files, num_source_files);
    size_t total_decls_count = 0;
    for (size_t i = 0; i < num_source_files; i++) {
        const char* input_file_contents = NULL;
        bool loaded_from_disk = false;

        if (i < num_builtin_sources_files) {
            input_file_contents = builtin_source_files[i];
        } else {
            const char* input_filename = read_list(const char*, input_files)[i - num_builtin_sources_files];
            input_file_contents = read_file(input_filename);
            if ((void*)input_file_contents == NULL) {
                error_print("file does not exist\n");
                exit(InputFileDoesNotExist);
            }
            loaded_from_disk = true;
        }

        info_print("Parsing: \n%s\n", input_file_contents);
        ParserConfig pconfig = {
            .front_end = true
        };
        const Node* parsed_file = parse(pconfig, input_file_contents, arena);
        parsed_files[i] = parsed_file;
        assert(parsed_file->tag == Root_TAG);
        total_decls_count += parsed_file->payload.root.declarations.count;

        if (loaded_from_disk)
            free((void*)input_file_contents);
    }

    destroy_list(input_files);

    // Merge all declarations into a giant program
    const Node** all_decls = malloc(sizeof(const Node*) * total_decls_count);
    size_t num_decl = 0;
    for (size_t i = 0; i < num_source_files; i++) {
        const Node* parsed_file = parsed_files[i];
        for (size_t j = 0; j < parsed_file->payload.root.declarations.count; j++)
            all_decls[num_decl++] = parsed_file->payload.root.declarations.nodes[j];
    }
    const Node* program = root(arena, (Root) {
        .declarations = nodes(arena, num_decl, all_decls)
    });
    free(all_decls);

    info_print("Parsed program successfully: \n");
    print_node(program);

    CompilationResult result = run_compiler_passes(&config, &arena, &program);
    if (result != CompilationNoError) {
        error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }

    if (cfg_output) {
        FILE* cfg_output_f = fopen(cfg_output, "wb");
        assert(cfg_output_f);
        dump_cfg(cfg_output_f, program);
        fclose(cfg_output_f);
    }

    info_print("Emitting final result ... \n");
    FILE *output = fopen(output_filename, "wb");
    emit_spirv(&config, arena, program, output);
    fclose(output);
    info_print("Done\n");

    destroy_arena(arena);
    return NoError;
}
