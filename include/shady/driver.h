#ifndef SHADY_DRIVER_H
#define SHADY_DRIVER_H

#include "shady/ir/base.h"
#include "shady/config.h"

#include "shady/be/c.h"
#include "shady/be/spirv.h"

struct List;

typedef enum {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist = 4,
    InputFileIOError,
    MissingDumpCfgArg,
    MissingDumpIrArg,
    IncorrectLogLevel = 16,
    InvalidTarget,
    ClangInvocationFailed,
} ShadyErrorCodes;

typedef enum {
    SrcShadyIR,
    SrcSlim,
    SrcSPIRV,
    SrcLLVM,
} SourceLanguage;

SourceLanguage shd_driver_guess_source_language(const char* filename);
ShadyErrorCodes shd_driver_load_source_file(const CompilerConfig* config, SourceLanguage lang, size_t len, const char* file_contents, String name, Module** mod);
ShadyErrorCodes shd_driver_load_source_file_from_filename(const CompilerConfig* config, const char* filename, String name, Module** mod);

typedef enum {
    TgtAuto,
    TgtC,
    TgtSPV,
    TgtGLSL,
    TgtISPC,
} CodegenTarget;

CodegenTarget shd_guess_target(const char* filename);

void shd_pack_remaining_args(int* pargc, char** argv);

// parses 'common' arguments such as log level etc
void shd_parse_common_args(int* pargc, char** argv);
// parses compiler pipeline options
void shd_parse_compiler_config_args(CompilerConfig* config, int* pargc, char** argv);
// parses whatever starts with '-'
void shd_driver_parse_unknown_options(struct List* list, int* pargc, char** argv);
// parses the remaining arguments into a list of files
void shd_driver_parse_input_files(struct List* list, int* pargc, char** argv);

typedef struct {
    CompilerConfig config;
    CodegenTarget target;
    struct {
        CTargetConfig c;
        SPIRVTargetConfig spirv;
    } target_config;
    struct List* input_filenames;
    const char*     output_filename;
    const char* shd_output_filename;
    const char* cfg_output_filename;
    const char* loop_tree_output_filename;
} DriverConfig;

DriverConfig shd_default_driver_config(void);
void shd_destroy_driver_config(DriverConfig* config);

void shd_parse_driver_args(DriverConfig* args, int* pargc, char** argv);

ShadyErrorCodes shd_driver_load_source_files(DriverConfig* args, Module* mod);
ShadyErrorCodes shd_driver_compile(DriverConfig* args, Module* mod);

typedef enum CompilationResult_ {
    CompilationNoError
} CompilationResult;

// CompilationResult shd_run_compiler_passes(CompilerConfig* config, Module** pmod);

#endif
