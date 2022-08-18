#ifndef SHADY_RUNTIME_H
#define SHADY_RUNTIME_H

#include <stdbool.h>

typedef struct {
    bool use_validation;
    bool dump_spv;
} RuntimeConfig;

typedef struct Runtime_ Runtime;
typedef struct Program_ Program;

Runtime* initialize_runtime(RuntimeConfig config);
void shutdown_runtime(Runtime*);

Program* load_program(Runtime*, const char* program_src);
void launch_kernel(Program*, int dimx, int dimy, int dimz, int extra_args_count, void** extra_args);

#endif
