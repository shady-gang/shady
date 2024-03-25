#include "cuda_runtime_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include <string.h>

static void shutdown_cuda_runtime(CudaBackend* b) {

}

static const char* cuda_device_get_name(CudaDevice* device) { return device->name; }

static void cuda_device_cleanup(CudaDevice* device) {

}

static CudaDevice* create_cuda_device(CudaBackend* b, int ordinal) {
    CUdevice handle;
    CHECK_CUDA(cuDeviceGet(&handle, ordinal), return NULL);
    CudaDevice* device = calloc(sizeof(CudaDevice), 1);
    *device = (CudaDevice) {
        .base = {
            .get_name = (const char*(*)(Device*)) cuda_device_get_name,
            .cleanup = (void(*)(Device*)) cuda_device_cleanup,
            .allocate_buffer = (Buffer* (*)(Device*, size_t)) shd_cuda_allocate_buffer,
            .can_import_host_memory = (bool (*)(Device*)) shd_cuda_can_import_host_memory,
            .import_host_memory_as_buffer = (Buffer* (*)(Device*, void*, size_t)) shd_cuda_import_host_memory,
        },
        .handle = handle,
    };
    CHECK_CUDA(cuDeviceGetName(device->name, 255, handle), goto dealloc_and_return_null);
    return device;

    dealloc_and_return_null:
    free(device);
    return NULL;
}

static bool probe_cuda_devices(CudaBackend* b) {
    int count;
    CHECK_CUDA(cuDeviceGetCount(&count), return false);
    for (size_t i = 0; i < count; i++) {
        CudaDevice* device = create_cuda_device(b, i);
        if (!device)
            continue;
        append_list(CudaDevice*, b->base.runtime->devices, device);
    }
    return true;
}

Backend* initialize_cuda_backend(Runtime* base) {
    CudaBackend* backend = malloc(sizeof(CudaBackend));
    memset(backend, 0, sizeof(CudaBackend));
    backend->base = (Backend) {
        .runtime = base,
        .cleanup = (void(*)()) shutdown_cuda_runtime,
    };

    CHECK_CUDA(cuInit(0), goto init_fail_free);
    CHECK(probe_cuda_devices(backend), goto init_fail_free);
    info_print("Shady CUDA backend successfully initialized !\n");
    return &backend->base;

    init_fail_free:
    error_print("Failed to initialise the CUDA back-end.\n");
    free(backend);
    return NULL;
}