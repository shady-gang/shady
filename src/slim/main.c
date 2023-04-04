#include "shady/ir.h"
#include "shady/cli.h"

#include "list.h"

#include "log.h"
#include "portability.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    CompilerConfig config;
    // Configuration specific to the C emitter
    CEmitterConfig c_emitter_config;
    CodegenTarget target;
    struct List* input_filenames;
    const char*     output_filename;
    const char* shd_output_filename;
    const char* cfg_output_filename;
    const char* loop_tree_output_filename;
} SlimConfig;

static void parse_slim_arguments(SlimConfig* args, int* pargc, char** argv) {
    int argc = *pargc;

    bool help = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--output must be followed with a filename");
                exit(MissingOutputArg);
            }
            args->output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-cfg") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-cfg must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->cfg_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-loop-tree") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-loop-tree must be followed with a filename");
                exit(MissingDumpCfgArg);
            }
            args->loop_tree_output_filename = argv[i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc) {
                error_print("--dump-ir must be followed with a filename");
                exit(MissingDumpIrArg);
            }
            args->shd_output_filename = argv[i];
        } else if (strcmp(argv[i], "--target") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                goto invalid_target;
            else if (strcmp(argv[i], "c") == 0)
                args->target = TgtC;
            else if (strcmp(argv[i], "spirv") == 0)
                args->target = TgtSPV;
            else if (strcmp(argv[i], "glsl") == 0)
                args->target = TgtGLSL;
            else if (strcmp(argv[i], "ispc") == 0)
                args->target = TgtISPC;
            else
                goto invalid_target;
            argv[i] = NULL;
            continue;
            invalid_target:
            error_print("--target must be followed with a valid target (see help for list of targets)");
            exit(InvalidTarget);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help = true;
            continue;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    if (help) {
        error_print("Usage: slim source.slim\n");
        error_print("Available arguments: \n");
        error_print("  --target <c, glsl, ispc, spirv>           \n");
        error_print("  --output <filename>, -o <filename>        \n");
        error_print("  --dump-cfg <filename>                     Dumps the control flow graph of the final IR\n");
        error_print("  --dump-loop-tree <filename>\n");
        error_print("  --dump-ir <filename>                      Dumps the final IR\n");
    }

    pack_remaining_args(pargc, argv);
}

void dump_loop_trees(FILE* output, Module* mod);

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    IrArena* arena = new_ir_arena(default_arena_config());

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

    parse_slim_arguments(&args, &argc, argv);
    parse_common_args(&argc, argv);
    parse_compiler_config_args(&args.config, &argc, argv);
    parse_input_files(args.input_filenames, &argc, argv);

    if (entries_count_list(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    Module* mod = new_module(arena, "my_module");

    size_t num_source_files = entries_count_list(args.input_filenames);
    CompilationResult parse_result = parse_files(&args.config, num_source_files, read_list(const char*, args.input_filenames), NULL, mod);
    assert(parse_result == CompilationNoError);
    destroy_list(args.input_filenames);

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

    if (args.loop_tree_output_filename) {
        FILE* f = fopen(args.loop_tree_output_filename, "wb");
        assert(f);
        dump_loop_trees(f, mod);
        fclose(f);
        info_print("Loop tree dumped\n");
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
        args.c_emitter_config.config = &args.config;
        if (args.target == TgtAuto)
            args.target = guess_target(args.output_filename);
        FILE* f = fopen(args.output_filename, "wb");
        size_t output_size;
        char* output_buffer;
        switch (args.target) {
            case TgtAuto: SHADY_UNREACHABLE;
            case TgtSPV: emit_spirv(&args.config, mod, &output_size, &output_buffer); break;
            case TgtC:
                args.c_emitter_config.dialect = C;
                emit_c(args.c_emitter_config, mod, &output_size, &output_buffer);
                break;
            case TgtGLSL:
                args.c_emitter_config.dialect = GLSL;
                emit_c(args.c_emitter_config, mod, &output_size, &output_buffer);
                break;
            case TgtISPC:
                args.c_emitter_config.dialect = ISPC;
                emit_c(args.c_emitter_config, mod, &output_size, &output_buffer);
                break;
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
