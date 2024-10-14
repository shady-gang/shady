#include "shady/ir.h"
#include "shady/driver.h"
#include "shady/print.h"
#include "shady/be/c.h"
#include "shady/be/spirv.h"
#include "shady/be/dump.h"

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

SourceLanguage shd_driver_guess_source_language(const char* filename) {
    if (shd_string_ends_with(filename, ".ll") || shd_string_ends_with(filename, ".bc"))
        return SrcLLVM;
    else if (shd_string_ends_with(filename, ".spv"))
        return SrcSPIRV;
    else if (shd_string_ends_with(filename, ".slim"))
        return SrcSlim;
    else if (shd_string_ends_with(filename, ".slim"))
        return SrcShadyIR;

    shd_warn_print("unknown filename extension '%s', interpreting as Slim sourcecode by default.", filename);
    return SrcSlim;
}

ShadyErrorCodes shd_driver_load_source_file(const CompilerConfig* config, SourceLanguage lang, size_t len, const char* file_contents, String name, Module** mod) {
    switch (lang) {
        case SrcLLVM: {
#ifdef LLVM_PARSER_PRESENT
            bool ok = shd_parse_llvm(config, len, file_contents, name, mod);
            assert(ok);
#else
            assert(false && "LLVM front-end missing in this version");
#endif
            break;
        }
        case SrcSPIRV: {
#ifdef SPV_PARSER_PRESENT
            shd_parse_spirv(config, len, file_contents, name, mod);
#else
            assert(false && "SPIR-V front-end missing in this version");
#endif
            break;
        }
        case SrcShadyIR:
        case SrcSlim: {
            SlimParserConfig pconfig = {
                .front_end = lang == SrcSlim,
            };
            shd_debugvv_print("Parsing: \n%s\n", file_contents);
            *mod = shd_parse_slim_module(config, &pconfig, (const char*) file_contents, name);
        }
    }
    return NoError;
}

ShadyErrorCodes shd_driver_load_source_file_from_filename(const CompilerConfig* config, const char* filename, String name, Module** mod) {
    ShadyErrorCodes err;
    SourceLanguage lang = shd_driver_guess_source_language(filename);
    size_t len;
    char* contents;
    assert(filename);
    bool ok = shd_read_file(filename, &len, &contents);
    if (!ok) {
        shd_error_print("Failed to read file '%s'\n", filename);
        err = InputFileIOError;
        goto exit;
    }
    if (contents == NULL) {
        shd_error_print("file does not exist\n");
        err = InputFileDoesNotExist;
        goto exit;
    }
    err = shd_driver_load_source_file(config, lang, len, contents, name, mod);
    free((void*) contents);
    exit:
    return err;
}

ShadyErrorCodes shd_driver_load_source_files(DriverConfig* args, Module* mod) {
    if (shd_list_count(args->input_filenames) == 0) {
        shd_error_print("Missing input file. See --help for proper usage");
        return MissingInputArg;
    }

    size_t num_source_files = shd_list_count(args->input_filenames);
    for (size_t i = 0; i < num_source_files; i++) {
        Module* m;
        int err = shd_driver_load_source_file_from_filename(&args->config,
                                                            shd_read_list(const char*, args->input_filenames)[i],
                                                            shd_read_list(const char*, args->input_filenames)[i], &m);
        if (err)
            return err;
        shd_module_link(mod, m);
        shd_destroy_ir_arena(shd_module_get_arena(m));
    }

    return NoError;
}

ShadyErrorCodes shd_driver_compile(DriverConfig* args, Module* mod) {
    shd_debugv_print("Parsed program successfully: \n");
    shd_log_module(DEBUGV, &args->config, mod);

    CompilationResult result = shd_run_compiler_passes(&args->config, &mod);
    if (result != CompilationNoError) {
        shd_error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }
    shd_debug_print("Ran all passes successfully\n");
    shd_log_module(DEBUG, &args->config, mod);

    if (args->cfg_output_filename) {
        FILE* f = fopen(args->cfg_output_filename, "wb");
        assert(f);
        shd_dump_cfgs(f, mod);
        fclose(f);
        shd_debug_print("CFG dumped\n");
    }

    if (args->loop_tree_output_filename) {
        FILE* f = fopen(args->loop_tree_output_filename, "wb");
        assert(f);
        shd_dump_loop_trees(f, mod);
        fclose(f);
        shd_debug_print("Loop tree dumped\n");
    }

    if (args->shd_output_filename) {
        FILE* f = fopen(args->shd_output_filename, "wb");
        assert(f);
        size_t output_size;
        char* output_buffer;
        shd_print_module_into_str(mod, &output_buffer, &output_size);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
        shd_debug_print("IR dumped\n");
    }

    if (args->output_filename) {
        if (args->target == TgtAuto)
            args->target = shd_guess_target(args->output_filename);
        FILE* f = fopen(args->output_filename, "wb");
        size_t output_size;
        char* output_buffer;
        switch (args->target) {
            case TgtAuto: SHADY_UNREACHABLE;
            case TgtSPV: emit_spirv(&args->config, mod, &output_size, &output_buffer, NULL); break;
            case TgtC:
                args->c_emitter_config.dialect = CDialect_C11;
                shd_emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtGLSL:
                args->c_emitter_config.dialect = CDialect_GLSL;
                shd_emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtISPC:
                args->c_emitter_config.dialect = CDialect_ISPC;
                shd_emit_c(&args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
        }
        shd_debug_print("Wrote result to %s\n", args->output_filename);
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
    }
    shd_destroy_ir_arena(shd_module_get_arena(mod));
    return NoError;
}
