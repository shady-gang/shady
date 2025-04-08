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
ShadyErrorCodes shd_driver_load_source_file(const CompilerConfig*, const TargetConfig*, SourceLanguage lang, size_t len, const char* file_contents, String name, Module** mod);
ShadyErrorCodes shd_driver_load_source_file_from_filename(const CompilerConfig*, const TargetConfig*, const char* filename, String name, Module** mod);

typedef enum {
    TgtAuto,
    TgtSPV,
    TgtC,
    TgtGLSL,
    TgtISPC,
    TgtCUDA,
} CodegenTarget;

typedef enum {
    BackendNone,
    BackendC,
    BackendSPV,
} BackendType;

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
    CodegenTarget target_type;
    BackendType backend_type;
    struct {
        CTargetConfig c;
        SPIRVTargetConfig spirv;
    } backend_config;
    struct List* input_filenames;
    const char*     output_filename;
    const char* shd_output_filename;
    const char* cfg_output_filename;
    const char* loop_tree_output_filename;

    struct {
        String entry_point;
        ExecutionModel execution_model;
    } specialization;
} DriverConfig;

DriverConfig shd_default_driver_config(void);
void shd_destroy_driver_config(DriverConfig* config);

void shd_parse_driver_args(DriverConfig* args, int* pargc, char** argv);

/// Populates the 'target' field of DriverConfig with defaults that match the driver options (output file etc)
void shd_driver_configure_target(TargetConfig*, DriverConfig* driver_config);
void shd_driver_configure_defaults_for_target(TargetConfig*, CodegenTarget);

/// Parses additional target configuration
void shd_parse_target_args(TargetConfig* target, int* pargc, char** argv);

ShadyErrorCodes shd_driver_load_source_files(const CompilerConfig* config, const TargetConfig* target_config, struct List* input_filenames, Module* mod);
ShadyErrorCodes shd_driver_compile(DriverConfig* args, const TargetConfig*, Module* mod);

typedef enum CompilationResult_ {
    CompilationNoError
} CompilationResult;

#endif
