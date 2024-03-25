#include "cuda_runtime_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"

#include <string.h>

static void shutdown_cuda_runtime(CudaBackend* b) {

}

static const char* cuda_device_get_name(CudaDevice* device) { return device->name; }

static void cuda_device_cleanup(CudaDevice* device) {
    size_t i = 0;
    CudaKernel* kernel;
    while (dict_iter(device->specialized_programs, &i, NULL, &kernel)) {
        shd_cuda_destroy_specialized_kernel(kernel);
    }
    destroy_dict(device->specialized_programs);
}

bool cuda_command_wait(CudaCommand* command) {
    CHECK_CUDA(cuCtxSynchronize(), return false);
    return true;
}

CudaCommand* shd_cuda_launch_kernel(CudaDevice* device, Program* p, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args) {
    CudaKernel* kernel = shd_cuda_get_specialized_program(device, p, entry_point);

    CudaCommand* cmd = calloc(sizeof(CudaCommand), 1);
    *cmd = (CudaCommand) {
        .base = {
            .wait_for_completion = (bool(*)(Command*)) cuda_command_wait
        }
    };
    ArenaConfig final_config = get_arena_config(get_module_arena(kernel->final_module));
    unsigned int gx = final_config.specializations.workgroup_size[0];
    unsigned int gy = final_config.specializations.workgroup_size[1];
    unsigned int gz = final_config.specializations.workgroup_size[2];
    CHECK_CUDA(cuLaunchKernel(kernel->entry_point_function, dimx, dimy, dimz, gx, gy, gz, 0, 0, args, NULL), return NULL);
    return cmd;
}

static KeyHash hash_spec_program_key(SpecProgramKey* ptr) {
    return hash_murmur(ptr, sizeof(SpecProgramKey));
}

static bool cmp_spec_program_keys(SpecProgramKey* a, SpecProgramKey* b) {
    return memcmp(a, b, sizeof(SpecProgramKey)) == 0;
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
            .launch_kernel = (Command*(*)(Device*, Program*, String, int, int, int, int, void**)) shd_cuda_launch_kernel,
        },
        .handle = handle,
        .specialized_programs = new_dict(SpecProgramKey, CudaKernel*, (HashFn) hash_spec_program_key, (CmpFn) cmp_spec_program_keys),
    };
    CHECK_CUDA(cuDeviceGetName(device->name, 255, handle), goto dealloc_and_return_null);
    CHECK_CUDA(cuCtxCreate(&device->context, 0, handle), goto dealloc_and_return_null);
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
        b->num_devices++;
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
    info_print("Shady CUDA backend successfully initialized, found %d devices\n", backend->num_devices);
    return &backend->base;

    init_fail_free:
    error_print("Failed to initialise the CUDA back-end.\n");
    free(backend);
    return NULL;
}