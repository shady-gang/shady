#ifndef SHADY_CLI
#define SHADY_CLI

#include "shady/ir.h"

enum ShadyErrorCodes {
    NoError,
    MissingInputArg,
    MissingOutputArg,
    InputFileDoesNotExist = 4,
    MissingDumpCfgArg,
    MissingDumpIrArg,
    IncorrectLogLevel = 16,
    InvalidTarget,
};

typedef enum {
    TgtAuto, TgtC, TgtSPV, TgtGLSL, TgtISPC,
} CodegenTarget;

struct List;

CodegenTarget guess_target(const char* filename);

void pack_remaining_args(int* pargc, char** argv);

// parses 'common' arguments such as log level etc
void parse_common_args(int* pargc, char** argv);
// parses compiler pipeline options
void parse_compiler_config_args(CompilerConfig*, int* pargc, char** argv);
// parses the remaining arguments into a list of files
void parse_input_files(struct List*, int* pargc, char** argv);

#endif
