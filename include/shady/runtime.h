#ifndef SHADY_RUNTIME_H
#define SHADY_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    bool use_validation;
    bool dump_spv;
    bool allow_no_devices;
} RuntimeConfig;

typedef struct Runtime_  Runtime;
typedef struct Device_   Device;
typedef struct Program_  Program;
typedef struct Dispatch_ Dispatch;
typedef struct Buffer_   Buffer;

Runtime* initialize_runtime(RuntimeConfig config);
void shutdown_runtime(Runtime*);

size_t device_count(Runtime*);
Device* get_device(Runtime*, size_t i);
const char* get_device_name(Device*);

Device* get_an_device(Runtime*);

Program* load_program(Runtime*, const char* program_src);
Dispatch* launch_kernel(Program*, Device*, int dimx, int dimy, int dimz, int args_count, void** args);
bool wait_completion(Dispatch*);

Buffer* allocate_buffer_device(Device*, size_t);
Buffer* import_buffer_host(Device*, void*, size_t);
void destroy_buffer(Buffer*);

uint64_t get_buffer_pointer(Buffer* buf);

bool copy_into_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size);
bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size);

#endif
