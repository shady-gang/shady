#ifndef SHADY_CUDA_RUNTIME_PRIVATE_H
#define SHADY_CUDA_RUNTIME_PRIVATE_H

#include "../runtime_private.h"

#include <cuda.h>
#include <nvrtc.h>

#define CHECK_NVRTC(x, failure_handler) { nvrtcResult the_result_ = x; if (the_result_ != NVRTC_SUCCESS) { const char* msg = nvrtcGetErrorString(the_result_); error_print(#x " failed (%s)\n", msg); failure_handler; } }
#define CHECK_CUDA(x, failure_handler) { CUresult the_result_ = x; if (the_result_ != CUDA_SUCCESS) { const char* msg; cuGetErrorName(the_result_, &msg); error_print(#x " failed (%s)\n", msg); failure_handler; } }

typedef struct {
    Program* base;
    String entry_point;
} SpecProgramKey;

typedef struct CudaBackend_ {
    Backend base;
} CudaBackend;

typedef struct {
    Device base;
    CUdevice handle;
    char name[256];
    struct Dict* specialized_programs;
} CudaDevice;

typedef struct {
    Buffer base;
    size_t size;
    CUdeviceptr device_ptr;
    void* host_ptr;
    bool is_allocated;
    bool is_imported;
} CudaBuffer;

typedef struct {
    Command base;
} CudaCommand;

typedef struct {
    SpecProgramKey key;
    CudaDevice* device;
    Module* final_module;
    struct {
        char* cuda_code;
        size_t cuda_code_size;
    };
    struct {
        char* ptx;
        size_t ptx_size;
    };
    CUmodule cuda_module;
    CUfunction entry_point_function;
} CudaKernel;

CudaBuffer* shd_cuda_allocate_buffer(CudaDevice*, size_t size);
CudaBuffer* shd_cuda_import_host_memory(CudaDevice*, void* host_ptr, size_t size);
bool shd_cuda_can_import_host_memory(CudaDevice*);

CudaKernel* shd_cuda_get_specialized_program(CudaDevice*, Program*, String ep);
bool shd_cuda_destroy_specialized_kernel(CudaKernel*);

#endif
