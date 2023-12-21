#include "runtime_private.h"

#include "log.h"
#include "list.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

Runtime* initialize_runtime(RuntimeConfig config) {
    Runtime* runtime = malloc(sizeof(Runtime));
    memset(runtime, 0, sizeof(Runtime));
    runtime->config = config;
    runtime->backends = new_list(Backend*);
    runtime->devices = new_list(Device*);
    runtime->programs = new_list(Program*);

#if VK_BACKEND_PRESENT
    Backend* vk_backend = initialize_vk_backend(runtime);
    CHECK(vk_backend, goto init_fail_free);
    append_list(Backend*, runtime->backends, vk_backend);
#endif
    info_print("Shady runtime successfully initialized !\n");
    return runtime;

    init_fail_free:
    error_print("Failed to initialise the runtime.\n");
    free(runtime);
    return NULL;
}

void shutdown_runtime(Runtime* runtime) {
    if (!runtime) return;

    // TODO force wait outstanding dispatches ?
    for (size_t i = 0; i < entries_count_list(runtime->devices); i++) {
        Device* dev = read_list(Device*, runtime->devices)[i];
        dev->cleanup(dev);
    }
    destroy_list(runtime->devices);

    for (size_t i = 0; i < entries_count_list(runtime->programs); i++) {
        unload_program(read_list(Program*, runtime->programs)[i]);
    }
    destroy_list(runtime->programs);

    for (size_t i = 0; i < entries_count_list(runtime->backends); i++) {
        Backend* bk = read_list(Backend*, runtime->backends)[i];
        bk->cleanup(bk);
    }
    free(runtime);
}

size_t device_count(Runtime* r) {
    return entries_count_list(r->devices);
}

Device* get_device(Runtime* r, size_t i) {
    assert(i < device_count(r));
    return read_list(Device*, r->devices)[i];
}

Device* get_an_device(Runtime* r) {
    assert(device_count(r) > 0);
    return get_device(r, 0);
}

// Virtual functions ...

const char* get_device_name(Device* d) { return d->get_name(d); }

Command* launch_kernel(Program* p, Device* d, const char* entry_point, int dimx, int dimy, int dimz, int args_count, void** args) {
    return d->launch_kernel(d, p, entry_point, dimx, dimy, dimz, args_count, args);
}

bool wait_completion(Command* cmd) { return cmd->wait_for_completion(cmd); }

bool can_import_host_memory(Device* device) { return device->can_import_host_memory(device); }

Buffer* allocate_buffer_device(Device* device, size_t bytes) { return device->allocate_buffer(device, bytes); }
Buffer* import_buffer_host(Device* device, void* ptr, size_t bytes) { return device->import_host_memory_as_buffer(device, ptr, bytes); }

void destroy_buffer(Buffer* buf) { buf->destroy(buf); };

void* get_buffer_host_pointer(Buffer* buf) { return buf->get_host_ptr(buf); }
uint64_t get_buffer_device_pointer(Buffer* buf) { return buf->get_device_ptr(buf); };

bool copy_to_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size) {
    return dst->copy_into(dst, buffer_offset, src, size);
}

bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size) {
    return src->copy_from(src, buffer_offset, dst, size);
}
