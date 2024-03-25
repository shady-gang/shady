#ifndef SHADY_CUDA_RUNTIME_PRIVATE_H
#define SHADY_CUDA_RUNTIME_PRIVATE_H

#include "../runtime_private.h"

#include <cuda.h>
#include <nvrtc.h>

#define CHECK_NVRTC(x, failure_handler) { CUresult the_result_ = x; if (the_result_ != NVRTC_SUCCESS) { const char* msg; nvrtcGetErrorString(the_result_, &msg); error_print(#x " failed (%s)\n", msg); failure_handler; } }
#define CHECK_CUDA(x, failure_handler) { CUresult the_result_ = x; if (the_result_ != CUDA_SUCCESS) { const char* msg; cuGetErrorName(the_result_, &msg); error_print(#x " failed (%s)\n", msg); failure_handler; } }

typedef struct CudaBackend_ {
    Backend base;
} CudaBackend;

typedef struct {
    Device base;
    CUdevice handle;
    char name[256];
} CudaDevice;

typedef struct {
    Buffer base;
    size_t size;
    CUdeviceptr device_ptr;
    void* host_ptr;
    bool is_allocated;
    bool is_imported;
} CudaBuffer;

CudaBuffer* shd_cuda_allocate_buffer(CudaDevice*, size_t size);
CudaBuffer* shd_cuda_import_host_memory(CudaDevice*, void* host_ptr, size_t size);
bool shd_cuda_can_import_host_memory(CudaDevice*);

#endif
