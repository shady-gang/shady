#include "shady/ir.h"
#include "shady/driver.h"

#include "shady/be/c.h"
#include "shady/be/spirv.h"
#include "shady/be/dump.h"

#include "print.h"

#include "../frontend/slim/parser.h"

#include "list.h"
#include "util.h"
#include "log.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef LLVM_PARSER_PRESENT
#include "../frontend/llvm/l2s.h"
#endif

#ifdef SPV_PARSER_PRESENT
#include "../frontend/spirv/s2s.h"
#endif

#pragma GCC diagnostic error "-Wswitch"

SourceLanguage guess_source_language(const char* filename) {
    if (string_ends_with(filename, ".ll") || string_ends_with(filename, ".bc"))
        return SrcLLVM;
    else if (string_ends_with(filename, ".spv"))
        return SrcSPIRV;
    else if (string_ends_with(filename, ".slim"))
        return SrcSlim;
    else if (string_ends_with(filename, ".slim"))
        return SrcShadyIR;

    warn_print("unknown filename extension '%s', interpreting as Slim sourcecode by default.", filename);
    return SrcSlim;
}

ShadyErrorCodes driver_load_source_file(const CompilerConfig* config, SourceLanguage lang, size_t len, const char* file_contents, String name, Module** mod) {
    switch (lang) {
        case SrcLLVM: {
#ifdef LLVM_PARSER_PRESENT
            bool ok = parse_llvm_into_shady(config, len, file_contents, name, mod);
            assert(ok);
#else
            assert(false && "LLVM front-end missing in this version");
#endif
            break;
        }
        case SrcSPIRV: {
#ifdef SPV_PARSER_PRESENT
            parse_spirv_into_shady(config, len, file_contents, name, mod);
#else
            assert(false && "SPIR-V front-end missing in this version");
#endif
            break;
        }
        case SrcShadyIR:
        case SrcSlim: {
            ParserConfig pconfig = {
                .front_end = lang == SrcSlim,
            };
            debugvv_print("Parsing: \n%s\n", file_contents);
            *mod = parse_slim_module(config, pconfig, (const char*) file_contents, name);
        }
    }
    return NoError;
}

ShadyErrorCodes driver_load_source_file_from_filename(const CompilerConfig* config, const char* filename, String name, Module** mod) {
    ShadyErrorCodes err;
    SourceLanguage lang = guess_source_language(filename);
    size_t len;
    char* contents;
    assert(filename);
    bool ok = read_file(filename, &len, &contents);
    if (!ok) {
        error_print("Failed to read file '%s'\n", filename);
        err = InputFileIOError;
        goto exit;
    }
    if (contents == NULL) {
        error_print("file does not exist\n");
        err = InputFileDoesNotExist;
        goto exit;
    }
    err = driver_load_source_file(config, lang, len, contents, name, mod);
    free((void*) contents);
    exit:
    return err;
}

ShadyErrorCodes driver_load_source_files(DriverConfig* args, Module* mod) {
    if (entries_count_list(args->input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        return MissingInputArg;
    }

    size_t num_source_files = entries_count_list(args->input_filenames);
    for (size_t i = 0; i < num_source_files; i++) {
        Module* m;
        int err = driver_load_source_file_from_filename(&args->config, read_list(const char*, args->input_filenames)[i], read_list(const char*, args->input_filenames)[i], &m);
        if (err)
            return err;
        link_module(mod, m);
        destroy_ir_arena(get_module_arena(m));
    }

    return NoError;
}

ShadyErrorCodes driver_compile(DriverConfig* args, Module* mod) {
    debugv_print("Parsed program successfully: \n");
    log_module(DEBUGV, &args->config, mod);

    CompilationResult result = run_compiler_passes(&args->config, &mod);
    if (result != CompilationNoError) {
        error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }
    debug_print("Ran all passes successfully\n");
    log_module(DEBUG, &args->config, mod);

    if (args->cfg_output_filename) {
        FILE* f = fopen(args->cfg_output_filename, "wb");
        assert(f);
        dump_cfgs(f, mod);
        fclose(f);
        debug_print("CFG dumped\n");
    }

    if (args->loop_tree_output_filename) {
        FILE* f = fopen(args->loop_tree_output_filename, "wb");
        assert(f);
        dump_loop_trees(f, mod);
        fclose(f);
        debug_print("Loop tree dumped\n");
    }

    if (args->shd_output_filename) {
        FILE* f = fopen(args->shd_output_filename, "wb");
        assert(f);
        size_t output_size;
        char* output_buffer;
        print_module_into_str(mod, &output_buffer, &output_size);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
        debug_print("IR dumped\n");
    }

    if (args->output_filename) {
        if (args->target == TgtAuto)
            args->target = guess_target(args->output_filename);
        FILE* f = fopen(args->output_filename, "wb");
        size_t output_size;
        char* output_buffer;
        switch (args->target) {
            case TgtAuto: SHADY_UNREACHABLE;
            case TgtSPV: emit_spirv(&args->config, mod, &output_size, &output_buffer, NULL); break;
            case TgtC:
                args->c_emitter_config.dialect = CDialect_C11;
                emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtGLSL:
                args->c_emitter_config.dialect = CDialect_GLSL;
                emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtISPC:
                args->c_emitter_config.dialect = CDialect_ISPC;
                emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
        }
        debug_print("Wrote result to %s\n", args->output_filename);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
    }
    destroy_ir_arena(get_module_arena(mod));
    return NoError;
}
