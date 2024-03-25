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
typedef struct Command_  Command;
typedef struct Buffer_   Buffer;

Runtime* initialize_runtime(RuntimeConfig config);
void shutdown_runtime(Runtime*);

size_t device_count(Runtime*);
Device* get_device(Runtime*, size_t i);
Device* get_an_device(Runtime*);
const char* get_device_name(Device*);

typedef struct CompilerConfig_ CompilerConfig;
typedef struct Module_ Module;

Program* new_program_from_module(Runtime*, const CompilerConfig*, Module*);
Program* load_program(Runtime*, const CompilerConfig*, const char* program_src);
Program* load_program_from_disk(Runtime*, const CompilerConfig*, const char* path);

Command* launch_kernel(Program*, Device*, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args);
bool wait_completion(Command*);

Buffer* allocate_buffer_device(Device*, size_t);
bool can_import_host_memory(Device*);
Buffer* import_buffer_host(Device*, void*, size_t);
void destroy_buffer(Buffer*);

void* get_buffer_host_pointer(Buffer* buf);
uint64_t get_buffer_device_pointer(Buffer* buf);

bool copy_to_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size);
bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size);

#endif
