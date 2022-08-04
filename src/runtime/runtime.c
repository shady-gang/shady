#include "shady/runtime.h"

#include "vulkan/vulkan.h"

#include <stdlib.h>

struct Runtime_ {
    VkInstance instance;
};

Runtime* initialize_runtime() {
    Runtime* runtime = malloc(sizeof(Runtime));
    if (!vkCreateInstance(&(VkInstanceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo) {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pEngineName = "shady",
            .pApplicationName = "business",
            .pNext = NULL,
            .engineVersion = 1,
            .applicationVersion = 1,
            .apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0)
        },
        .flags = 0,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = (const char* []) {},
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .pNext = NULL
    }, NULL, &runtime->instance)) goto init_fail;

    return runtime;

    init_fail:
    free(runtime);
    return NULL;
}
