#include "shady/ir.h"

#include "list.h"

#include "log.h"
#include "portability.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

char* read_file(const char* filename);

enum SlimErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist,
    IncorrectLogLevel,
    MissingDumpCfgArg,
    MissingDumpIrArg,
    InvalidTarget,
};

typedef enum {
    TgtAuto, TgtC, TgtSPV, TgtGLSL,
} CodegenTarget;


static bool string_ends_with(const char* string, const char* suffix) {
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

static CodegenTarget guess_target(const char* filename) {
    if (string_ends_with(filename, ".c"))
        return TgtC;
    else if (string_ends_with(filename, "glsl"))
        return TgtGLSL;
    else if (string_ends_with(filename, "spirv"))
        return TgtSPV;
    else if (string_ends_with(filename, "spv"))
        return TgtSPV;
    error_print("No target has been specified, and output filename '%s' did not allow guessing the right one\n");
    exit(InvalidTarget);
}

typedef struct {
    CompilerConfig config;
    CodegenTarget target;
    struct List* input_filenames;
    const char*     output_filename;
    const char* shd_output_filename;
    const char* cfg_output_filename;
} SlimConfig;

static void process_arguments(int argc, const char** argv, SlimConfig* args) {
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
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            args->output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->cfg_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            i++;
            if (i == argc) {
                error_print("--dump-ir must be followed with a filename");
                exit(MissingDumpIrArg);
            }
            args->shd_output_filename = argv[i];
        } else if (strcmp(argv[i], "--target") == 0) {
            i++;
            if (i == argc)
                goto invalid_target;
            else if (strcmp(argv[i], "c") == 0)
                args->target = TgtC;
            else if (strcmp(argv[i], "spirv") == 0)
                args->target = TgtSPV;
            else
                goto invalid_target;
            continue;
            invalid_target:
            error_print("--target must be followed with a valid target (see help for list of targets)");
            exit(InvalidTarget);
        } else if (strcmp(argv[i], "--print-builtin") == 0) {
            args->config.logging.skip_builtin = false;
            i++;
        } else if (strcmp(argv[i], "--print-generated") == 0) {
            args->config.logging.skip_generated = false;
            i++;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            i++;
        } else {
            append_list(const char*, args->input_filenames, argv[i]);
        }
    }

    if (entries_count_list(args->input_filenames) == 0 || help) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --log-level [debug, info, warn, error]\n");
        error_print("  --target c spirv\n");
        error_print("  --output <filename>, -o <filename>\n");
        error_print("  --dump-cfg <filename>\n");
        error_print("  --dump-ir <filename>\n");
        error_print("  --print-builtin\n");
        error_print("  --print-generated\n");
        exit(help ? 0 : MissingInputArg);
    }
}

int main(int argc, const char** argv) {
    platform_specific_terminal_init_extras();

    IrArena* arena = new_ir_arena((ArenaConfig) {
        .check_types = false
    });

    SlimConfig args = {
        .config = default_compiler_config(),
        .target = TgtAuto,
        .input_filenames = new_list(const char*),
        .output_filename = NULL,
        .cfg_output_filename = NULL,
        .shd_output_filename = NULL,
    };
    args.config.allow_frontend_syntax = true;

    // most of the time, we are not interested in seeing generated/builtin code in the debug output
    args.config.logging.skip_builtin = true;
    args.config.logging.skip_generated = true;

    process_arguments(argc, argv, &args);

    // Read the files
    size_t num_source_files = entries_count_list(args.input_filenames);
    LARRAY(const char*, read_files, num_source_files);
    for (size_t i = 0; i < num_source_files; i++) {
        const char* input_file_contents = read_file(read_list(const char*, args.input_filenames)[i]);
        if ((void*)input_file_contents == NULL) {
            error_print("file does not exist\n");
            exit(InputFileDoesNotExist);
        }
        read_files[i] = input_file_contents;
    }
    destroy_list(args.input_filenames);

    // Parse the lot
    Module* mod = new_module(arena, "my_module");
    CompilationResult parse_result = parse_files(&args.config, num_source_files, read_files, mod);
    assert(parse_result == CompilationNoError);

    // Free the read files
    for (size_t i = 0; i < num_source_files; i++)
        free((void*) read_files[i]);

    info_print("Parsed program successfully: \n");
    log_module(INFO, &args.config, mod);

    CompilationResult result = run_compiler_passes(&args.config, &mod);
    if (result != CompilationNoError) {
        error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }
    info_print("Ran all passes successfully\n");

    if (args.cfg_output_filename) {
        FILE* f = fopen(args.cfg_output_filename, "wb");
        assert(f);
        dump_cfg(f, mod);
        fclose(f);
        info_print("CFG dumped\n");
    }

    if (args.shd_output_filename) {
        FILE* f = fopen(args.shd_output_filename, "wb");
        assert(f);
        size_t output_size;
        char* output_buffer;
        print_module_into_str(mod, &output_buffer, &output_size);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
        info_print("IR dumped\n");
    }

    if (args.output_filename) {
        if (args.target == TgtAuto)
            args.target = guess_target(args.output_filename);
        FILE* f = fopen(args.output_filename, "wb");
        size_t output_size;
        char* output_buffer;
        switch (args.target) {
            case TgtAuto: SHADY_UNREACHABLE;
            case TgtSPV: emit_spirv(&args.config, mod, &output_size, &output_buffer); break;
            case TgtC: emit_c(&args.config, C, mod, &output_size, &output_buffer); break;
            case TgtGLSL: emit_c(&args.config, GLSL, mod, &output_size, &output_buffer); break;
        }
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
    }
    info_print("Done\n");

    destroy_ir_arena(arena);
    destroy_ir_arena(get_module_arena(mod));
    return NoError;
}
