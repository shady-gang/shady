#include "shady/ir.h"
#include "shady/driver.h"

#include "../shady/parser/parser.h"

#include "list.h"
#include "util.h"

#include "log.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef LLVM_PARSER_PRESENT
#include "../frontends/llvm/l2s.h"
#endif

#ifdef SPV_PARSER_PRESENT
#include "../frontends/spirv/s2s.h"
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

    warn_print("unknown filename extension '%s', interpreting as Slim sourcecode by default.");
    return SrcSlim;
}

ShadyErrorCodes parse_file(SourceLanguage lang, size_t len, const char* file_contents, Module* mod) {
    switch (lang) {
        case SrcLLVM: {
#ifdef LLVM_PARSER_PRESENT
            parse_llvm_into_shady(mod, len, file_contents);
#else
            assert(false && "LLVM front-end missing in this version");
#endif
            break;
        }
        case SrcSPIRV: {
#ifdef SPV_PARSER_PRESENT
            parse_spirv_into_shady(mod, len, file_contents);
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
            debugv_print("Parsing: \n%s\n", file_contents);
            parse_shady_ir(pconfig, (const char*) file_contents, mod);
        }
    }
    return NoError;
}

ShadyErrorCodes driver_compile(DriverConfig* args, Module* mod) {
    info_print("Parsed program successfully: \n");
    log_module(INFO, &args->config, mod);

    CompilationResult result = run_compiler_passes(&args->config, &mod);
    if (result != CompilationNoError) {
        error_print("Compilation pipeline failed, errcode=%d\n", (int) result);
        exit(result);
    }
    info_print("Ran all passes successfully\n");

    if (args->cfg_output_filename) {
        FILE* f = fopen(args->cfg_output_filename, "wb");
        assert(f);
        dump_cfg(f, mod);
        fclose(f);
        info_print("CFG dumped\n");
    }

    if (args->loop_tree_output_filename) {
        FILE* f = fopen(args->loop_tree_output_filename, "wb");
        assert(f);
        dump_loop_trees(f, mod);
        fclose(f);
        info_print("Loop tree dumped\n");
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
        info_print("IR dumped\n");
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
                args->c_emitter_config.dialect = C;
                emit_c(args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtGLSL:
                args->c_emitter_config.dialect = GLSL;
                emit_c(args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
            case TgtISPC:
                args->c_emitter_config.dialect = ISPC;
                emit_c(args->config, args->c_emitter_config, mod, &output_size, &output_buffer, NULL);
                break;
        }
        fwrite(output_buffer, output_size, 1, f);
        free((void*) output_buffer);
        fclose(f);
    }
    destroy_ir_arena(get_module_arena(mod));
    return NoError;
}
