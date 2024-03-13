#include "cuda_runtime_private.h"

#include "log.h"
#include "portability.h"

#include <string.h>

Backend* initialize_cuda_backend(Runtime* base) {
    CudaBackend* backend = malloc(sizeof(CudaBackend));
    memset(backend, 0, sizeof(CudaBackend));
    backend->base = (Backend) {
        .runtime = base,
        // .cleanup = (void(*)()) shutdown_vulkan_runtime,
    };

    // CHECK(initialize_vk_instance(backend), goto init_fail_free)
    // probe_vkr_devices(backend);
    info_print("Shady CUDA backend successfully initialized !\n");
    return &backend->base;

    init_fail_free:
    error_print("Failed to initialise the CUDA back-end.\n");
    free(backend);
    return NULL;
}