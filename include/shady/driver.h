#ifndef SHADY_CLI
#define SHADY_CLI

#include "shady/ir.h"

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

SourceLanguage guess_source_language(const char* filename);
ShadyErrorCodes driver_load_source_file(const CompilerConfig* config, SourceLanguage lang, size_t, const char* file_contents, String, Module** mod);
ShadyErrorCodes driver_load_source_file_from_filename(const CompilerConfig* config, const char* filename, String, Module** mod);

typedef enum {
    TgtAuto,
    TgtC,
    TgtSPV,
    TgtGLSL,
    TgtISPC,
} CodegenTarget;

CodegenTarget guess_target(const char* filename);

void cli_pack_remaining_args(int* pargc, char** argv);

// parses 'common' arguments such as log level etc
void cli_parse_common_args(int* pargc, char** argv);
// parses compiler pipeline options
void cli_parse_compiler_config_args(CompilerConfig*, int* pargc, char** argv);
// parses the remaining arguments into a list of files
void cli_parse_input_files(struct List*, int* pargc, char** argv);

typedef struct {
    CompilerConfig config;
    CEmitterConfig c_emitter_config;
    struct List* input_filenames;
    CodegenTarget target;
    const char*     output_filename;
    const char* shd_output_filename;
    const char* cfg_output_filename;
    const char* loop_tree_output_filename;
} DriverConfig;

DriverConfig default_driver_config();
void destroy_driver_config(DriverConfig*);

void cli_parse_driver_arguments(DriverConfig* args, int* pargc, char** argv);

ShadyErrorCodes driver_load_source_files(DriverConfig* args, Module* mod);
ShadyErrorCodes driver_compile(DriverConfig* args, Module* mod);

#endif
