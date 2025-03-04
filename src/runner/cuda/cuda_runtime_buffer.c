#include "cuda_runtime_private.h"

#include "log.h"
#include "portability.h"

static void cuda_destroy_buffer(CudaBuffer* buffer) {
    if (buffer->is_allocated)
    CHECK_CUDA(cuMemFree(buffer->device_ptr), {});
    if (buffer->is_imported)
    CHECK_CUDA(cuMemHostUnregister(buffer->host_ptr), {});
    free(buffer);
}

static uint64_t cuda_get_deviceptr(CudaBuffer* buffer) {
    return (uint64_t) buffer->device_ptr;
}

static void* cuda_get_host_ptr(CudaBuffer* buffer) {
    return (void*) buffer->host_ptr;
}

static bool cuda_copy_to_buffer_fallback(CudaBuffer* dst, size_t dst_offset, void* src, size_t size) {
    CHECK_CUDA(cuMemcpyHtoD(dst->device_ptr + dst_offset, src, size), return false);
    return true;
}

static bool cuda_copy_from_buffer_fallback(CudaBuffer* src, size_t src_offset, void* dst, size_t size) {
    CHECK_CUDA(cuMemcpyDtoH(dst, src->device_ptr + src_offset, size), return false);
    return true;
}

static CudaBuffer* new_buffer_common(size_t size) {
    CudaBuffer* buffer = calloc(sizeof(CudaBuffer), 1);
    *buffer = (CudaBuffer) {
        .base = {
            .backend_tag = CUDARuntimeBackend,
            .get_host_ptr = (void*(*)(Buffer*)) cuda_get_host_ptr,
            .get_device_ptr = (uint64_t(*)(Buffer*)) cuda_get_deviceptr,
            .destroy = (void(*)(Buffer*)) cuda_destroy_buffer,
            .copy_into = (bool(*)(Buffer*, size_t, void*, size_t)) cuda_copy_to_buffer_fallback,
            .copy_from = (bool(*)(Buffer*, size_t, void*, size_t)) cuda_copy_from_buffer_fallback,
        },
        .size = size,
    };
    return buffer;
}

CudaBuffer* shd_rt_cuda_allocate_buffer(CudaDevice* device, size_t size) {
    CUdeviceptr device_ptr;
    CHECK_CUDA(cuMemAlloc(&device_ptr, size), return NULL);
    CudaBuffer* buffer = new_buffer_common(size);
    buffer->is_allocated = true;
    buffer->device_ptr = device_ptr;
    // TODO: check the assumptions of unified virtual addressing
    buffer->host_ptr = (void*) device_ptr;
    return buffer;
}

CudaBuffer* shd_rt_cuda_import_host_memory(CudaDevice* device, void* host_ptr, size_t size) {
    CUdeviceptr device_ptr;
    CHECK_CUDA(cuMemHostRegister(host_ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP), return NULL);
    CHECK_CUDA(cuMemHostGetDevicePointer(&device_ptr, host_ptr, 0), return NULL);
    CudaBuffer* buffer = new_buffer_common(size);
    buffer->is_imported = true;
    buffer->device_ptr = device_ptr;
    // TODO: check the assumptions of unified virtual addressing
    buffer->host_ptr = (void*) host_ptr;
    return buffer;
}

bool shd_rt_cuda_can_import_host_memory(CudaDevice* d) {
    return true;
}
