#include "cuda_runner_private.h"
#include "shady/config.h"
#include "shady/ir/module.h"

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
    while (shd_dict_iter(device->specialized_programs, &i, NULL, &kernel)) {
        shd_cur_destroy_specialized_kernel(kernel);
    }
    shd_destroy_dict(device->specialized_programs);
}

static bool command_wait(CudaCommand* command) {
    CHECK_CUDA(cuCtxSynchronize(), return false);
    if (command->profiled_gpu_time) {
        cudaEventSynchronize(command->stop);
        float ms;
        cudaEventElapsedTime(&ms, command->start, command->stop);
        *command->profiled_gpu_time = (uint64_t) ((double) ms * 1000000);
    }
    return true;
}

static CudaCommand* launch_kernel(CudaDevice* device, Program* p, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions* options) {
    CudaKernel* kernel = shd_cur_get_specialized_program(device, p, entry_point);

    CudaCommand* cmd = calloc(sizeof(CudaCommand), 1);
    *cmd = (CudaCommand) {
        .base = {
            .wait_for_completion = (bool (*)(Command*)) command_wait
        }
    };

    if (options && options->profiled_gpu_time) {
        cmd->profiled_gpu_time = options->profiled_gpu_time;
        cudaEventCreate(&cmd->start);
        cudaEventCreate(&cmd->stop);
        cudaEventRecord(cmd->start, 0);
    }

    ArenaConfig final_config = *shd_get_arena_config(shd_module_get_arena(kernel->final_module));
    unsigned int gx = final_config.specializations.workgroup_size[0];
    unsigned int gy = final_config.specializations.workgroup_size[1];
    unsigned int gz = final_config.specializations.workgroup_size[2];
    CHECK_CUDA(cuLaunchKernel(kernel->entry_point_function, dimx, dimy, dimz, gx, gy, gz, 0, 0, args, NULL), return NULL);
    cudaEventRecord(cmd->stop, 0);
    return cmd;
}

static KeyHash hash_spec_program_key(SpecProgramKey* ptr) {
    return shd_hash(ptr, sizeof(SpecProgramKey));
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
            .backend = CUDARuntimeBackend,
            .get_name = (const char*(*)(Device*)) cuda_device_get_name,
            .cleanup = (void(*)(Device*)) cuda_device_cleanup,
            .allocate_buffer = (Buffer* (*)(Device*, size_t)) shd_cur_allocate_buffer,
            .can_import_host_memory = (bool (*)(Device*)) shd_cur_can_import_host_memory,
            .import_host_memory_as_buffer = (Buffer* (*)(Device*, void*, size_t)) shd_cur_import_host_memory,
            .launch_kernel = (Command* (*)(Device*, Program*, String, int, int, int, int, void**, ExtraKernelOptions*)) launch_kernel,
        },
        .handle = handle,
        .specialized_programs = shd_new_dict(SpecProgramKey, CudaKernel*, (HashFn) hash_spec_program_key, (CmpFn) cmp_spec_program_keys),
    };
    CHECK_CUDA(cuDeviceGetName(device->name, 255, handle), goto dealloc_and_return_null);
    CHECK_CUDA(cuDeviceGetAttribute(&device->cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device->handle), goto dealloc_and_return_null);
    CHECK_CUDA(cuDeviceGetAttribute(&device->cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device->handle), goto dealloc_and_return_null);
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
        shd_list_append(CudaDevice*, b->base.runner->devices, device);
    }
    return true;
}

Backend* shd_cur_init(Runner* base) {
    CudaBackend* backend = malloc(sizeof(CudaBackend));
    memset(backend, 0, sizeof(CudaBackend));
    backend->base = (Backend) {
        .runner = base,
        .backend_type = CUDARuntimeBackend,
        .cleanup = (void(*)()) shutdown_cuda_runtime,
    };

    CHECK_CUDA(cuInit(0), goto init_fail_free);
    CHECK(probe_cuda_devices(backend), goto init_fail_free);
    shd_info_print("Shady CUDA backend successfully initialized, found %d devices\n", backend->num_devices);
    return &backend->base;

    init_fail_free:
    shd_error_print("Failed to initialise the CUDA back-end.\n");
    free(backend);
    return NULL;
}