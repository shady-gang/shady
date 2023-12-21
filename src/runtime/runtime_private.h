#ifndef SHADY_RUNTIME_PRIVATE
#define SHADY_RUNTIME_PRIVATE
#include "shady/runtime.h"
#include "shady/ir.h"

#define CHECK(x, failure_handler) { if (!(x)) { error_print(#x " failed\n"); failure_handler; } }

// typedef struct SpecProgram_ SpecProgram;

struct Runtime_ {
    RuntimeConfig config;

    struct List* backends;
    struct List* devices;
    struct List* programs;
};

typedef struct Backend_ Backend;
struct Backend_ {
    Runtime* runtime;
    void (*cleanup)(Backend*);
};

struct Device_ {
    void (*cleanup)(Device*);
    String (*get_name)(Device*);

    Command* (*launch_kernel)(Device*, Program*, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args);
    Buffer* (*allocate_buffer)(Device*, size_t bytes);
    Buffer* (*import_host_memory_as_buffer)(Device*, void* base, size_t bytes);
    bool (*can_import_host_memory)(Device*);
};

struct Program_ {
    Runtime* runtime;
    const CompilerConfig* base_config;
    /// owns the module, may be NULL if module is owned by someone else
    IrArena* arena;
    Module* module;
};

struct Command_ {
    bool (*wait_for_completion)(Command*);
};

struct Buffer_ {
    void (*destroy)(Buffer*);
    void*    (*get_host_ptr)(Buffer*);
    uint64_t (*get_device_ptr)(Buffer*);

    bool (*copy_into)(Buffer* dst, size_t buffer_offset, void* src, size_t bytes);
    bool (*copy_from)(Buffer* src, size_t buffer_offset, void* dst, size_t bytes);
};

void unload_program(Program*);

Backend* initialize_vk_backend(Runtime*);
#endif
