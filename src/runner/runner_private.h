#ifndef SHADY_RUNTIME_PRIVATE
#define SHADY_RUNTIME_PRIVATE
#include "shady/runner.h"
#include "shady/ir.h"

// typedef struct SpecProgram_ SpecProgram;

struct Runner_ {
    RunnerConfig config;

    struct List* backends;
    struct List* devices;
    struct List* programs;
};

typedef enum {
    VulkanRuntimeBackend,
    CUDARuntimeBackend,
} RunnerBackend;

typedef struct Backend_ Backend;
struct Backend_ {
    Runner* runner;
    void (*cleanup)(Backend*);
};

struct Device_ {
    void (*cleanup)(Device*);
    String (*get_name)(Device*);

    Command* (*launch_kernel)(Device*, Program*, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions*);
    Buffer* (*allocate_buffer)(Device*, size_t bytes);
    Buffer* (*import_host_memory_as_buffer)(Device*, void* base, size_t bytes);
    bool (*can_import_host_memory)(Device*);
};

typedef struct {
    size_t num_args;
    const size_t* arg_offset;
    const size_t* arg_size;
    size_t args_size;
} ProgramParamsInfo;

struct Program_ {
    Runner* runtime;
    const CompilerConfig* base_config;
    /// owns the module, may be NULL if module is owned by someone else
    IrArena* arena;
    Module* module;
};

struct Command_ {
    bool (*wait_for_completion)(Command*);
};

struct Buffer_ {
    RunnerBackend backend_tag;
    void     (*destroy)(Buffer*);
    void*    (*get_host_ptr)(Buffer*);
    uint64_t (*get_device_ptr)(Buffer*);
    bool     (*copy_into)(Buffer* dst, size_t buffer_offset, void* src, size_t bytes);
    bool     (*copy_from)(Buffer* src, size_t buffer_offset, void* dst, size_t bytes);
};

void shd_rn_unload_program(Program* program);

Backend* shd_vkr_init(Runner* base);
Backend* shd_rt_initialize_cuda_backend(Runner*);

#endif
