#ifndef SHADY_RUNTIME_H
#define SHADY_RUNTIME_H

typedef struct Runtime_ Runtime;
typedef struct Program_ Program;

Runtime* initialize_runtime();
void shutdown_runtime(Runtime*);

Program* load_program(Runtime*, const char* program_src);
void launch_kernel(Program*, int dimx, int dimy, int dimz, int extra_args_count, void** extra_args);

#endif
