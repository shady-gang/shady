#ifndef SHADY_RUNTIME_H
#define SHADY_RUNTIME_H

#include <stdbool.h>

typedef struct {
    bool use_validation;
    bool dump_spv;
} RuntimeConfig;

typedef struct Runtime_  Runtime;
typedef struct Device_   Device;
typedef struct Program_  Program;
typedef struct Dispatch_ Dispatch;

Runtime* initialize_runtime(RuntimeConfig config);
void shutdown_runtime(Runtime*);

// TODO: API for enumerating devices
Device* initialize_device(Runtime*);

Program* load_program(Runtime*, const char* program_src);

Dispatch* launch_kernel(Program*, Device*, int dimx, int dimy, int dimz, int extra_args_count, void** extra_args);
bool wait_completion(Dispatch*);

#endif
