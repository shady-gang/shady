#include "runner_private.h"

#include "log.h"
#include "list.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

Runtime* shd_rt_initialize(RuntimeConfig config) {
    Runtime* runtime = malloc(sizeof(Runtime));
    memset(runtime, 0, sizeof(Runtime));
    runtime->config = config;
    runtime->backends = shd_new_list(Backend*);
    runtime->devices = shd_new_list(Device*);
    runtime->programs = shd_new_list(Program*);

#if VK_BACKEND_PRESENT
    Backend* vk_backend = shd_rt_initialize_vk_backend(runtime);
    CHECK(vk_backend, goto init_fail_free);
    shd_list_append(Backend*, runtime->backends, vk_backend);
#endif
#if CUDA_BACKEND_PRESENT
    Backend* cuda_backend = shd_rt_initialize_cuda_backend(runtime);
    CHECK(cuda_backend, goto init_fail_free);
    append_list(Backend*, runtime->backends, cuda_backend);
#endif

    shd_info_print("Shady runtime successfully initialized !\n");
    return runtime;

    init_fail_free:
    shd_error_print("Failed to initialise the runtime.\n");
    free(runtime);
    return NULL;
}

void shd_rt_shutdown(Runtime* runtime) {
    if (!runtime) return;

    // TODO force wait outstanding dispatches ?
    for (size_t i = 0; i < shd_list_count(runtime->devices); i++) {
        Device* dev = shd_read_list(Device*, runtime->devices)[i];
        dev->cleanup(dev);
    }
    shd_destroy_list(runtime->devices);

    for (size_t i = 0; i < shd_list_count(runtime->programs); i++) {
        shd_rt_unload_program(shd_read_list(Program*, runtime->programs)[i]);
    }
    shd_destroy_list(runtime->programs);

    for (size_t i = 0; i < shd_list_count(runtime->backends); i++) {
        Backend* bk = shd_read_list(Backend*, runtime->backends)[i];
        bk->cleanup(bk);
    }
    shd_destroy_list(runtime->backends);
    free(runtime);
}

size_t shd_rt_device_count(Runtime* r) {
    return shd_list_count(r->devices);
}

Device* shd_rt_get_device(Runtime* r, size_t i) {
    assert(i < shd_rt_device_count(r));
    return shd_read_list(Device*, r->devices)[i];
}

Device* shd_rt_get_an_device(Runtime* r) {
    assert(shd_rt_device_count(r) > 0);
    return shd_rt_get_device(r, 0);
}

// Virtual functions ...

const char* shd_rt_get_device_name(Device* d) { return d->get_name(d); }

Command* shd_rt_launch_kernel(Program* p, Device* d, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions* extra_options) {
    return d->launch_kernel(d, p, entry_point, dimx, dimy, dimz, args_count, args, extra_options);
}

bool shd_rt_wait_completion(Command* cmd) { return cmd->wait_for_completion(cmd); }

bool shd_rt_can_import_host_memory(Device* device) { return device->can_import_host_memory(device); }

Buffer* shd_rt_allocate_buffer_device(Device* device, size_t bytes) { return device->allocate_buffer(device, bytes); }
Buffer* shd_rt_import_buffer_host(Device* device, void* ptr, size_t bytes) { return device->import_host_memory_as_buffer(device, ptr, bytes); }

void shd_rt_destroy_buffer(Buffer* buf) { buf->destroy(buf); }

void* shd_rt_get_buffer_host_pointer(Buffer* buf) { return buf->get_host_ptr(buf); }
uint64_t shd_rt_get_buffer_device_pointer(Buffer* buf) { return buf->get_device_ptr(buf); }

bool shd_rt_copy_to_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size) { return dst->copy_into(dst, buffer_offset, src, size); }
bool shd_rt_copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size) { return src->copy_from(src, buffer_offset, dst, size); }
