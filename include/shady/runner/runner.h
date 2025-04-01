#ifndef SHADY_RUNNER_H
#define SHADY_RUNNER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    bool use_validation;
    bool dump_spv;
    bool allow_no_devices;
} RunnerConfig;

RunnerConfig shd_rn_default_config(void);
void shd_rn_cli_parse_config(RunnerConfig* config, int* pargc, char** argv);

typedef struct Runner_  Runner;
typedef struct Device_  Device;
typedef struct Program_ Program;
typedef struct Command_ Command;
typedef struct Buffer_  Buffer;

Runner* shd_rn_initialize(RunnerConfig config);
void shd_rn_shutdown(Runner* runtime);

size_t shd_rn_device_count(Runner* r);
Device* shd_rn_get_device(Runner* r, size_t i);
Device* shd_rn_get_an_device(Runner* r);
const char* shd_rn_get_device_name(Device* d);

typedef struct CompilerConfig_ CompilerConfig;
typedef struct Module_ Module;

Program* shd_rn_new_program_from_module(Runner* runtime, const CompilerConfig* base_config, Module* mod);

typedef struct {
    uint64_t* profiled_gpu_time;
} ExtraKernelOptions;

Command* shd_rn_launch_kernel(Program* p, Device* d, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions* extra_options);
bool shd_rn_wait_completion(Command* cmd);

Buffer* shd_rn_allocate_buffer_device(Device* device, size_t bytes);
bool shd_rn_can_import_host_memory(Device* device);
Buffer* shd_rn_import_buffer_host(Device* device, void* ptr, size_t bytes);
void shd_rn_destroy_buffer(Buffer* buf);

void* shd_rn_get_buffer_host_pointer(Buffer* buf);
uint64_t shd_rn_get_buffer_device_pointer(Buffer* buf);

bool shd_rn_copy_to_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size);
bool shd_rn_copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size);

#endif
