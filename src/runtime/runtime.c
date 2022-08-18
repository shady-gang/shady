#include "runtime_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

static VKAPI_ATTR VkBool32 VKAPI_CALL the_callback(SHADY_UNUSED VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, SHADY_UNUSED VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, SHADY_UNUSED void* pUserData) {
    warn_print("Validation says: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

static bool setup_debug_callback(Runtime* runtime) {
    CHECK_VK(runtime->instance_exts.debug_utils.vkCreateDebugUtilsMessengerEXT(runtime->instance, &(VkDebugUtilsMessengerCreateInfoEXT) {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = the_callback,
        .pUserData = NULL
    }, NULL, &runtime->debug_messenger), return false);
    return true;
}

static void obtain_instance_pointers(Runtime* runtime) {
    #define Y(fn_name) ext->fn_name = (PFN_##fn_name) vkGetInstanceProcAddr(runtime->instance, #fn_name);
    #define X(_, prefix, name, fns) \
        if (runtime->instance_exts.name.enabled) { \
            struct S_##name* ext = &runtime->instance_exts.name; \
            fns(Y) \
        }
    INSTANCE_EXTENSIONS(X)
    #undef Y
    #undef X
}

static bool initialize_vk_instance(Runtime* runtime) {
    uint32_t layers_count;
    CHECK_VK(vkEnumerateInstanceLayerProperties(&layers_count, NULL), return false);
    LARRAY(VkLayerProperties, layer_properties, layers_count);
    CHECK_VK(vkEnumerateInstanceLayerProperties(&layers_count, layer_properties), return false);

    uint32_t enabled_layers_count = 0;
    LARRAY(const char*, enabled_layers, layers_count);

    for (uint32_t i = 0; i < layers_count; i++) {
        VkLayerProperties* layer = &layer_properties[i];

        // Enable validation if the config says so
        if (runtime->config.use_validation && strcmp(layer->layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            info_print("Enabling validation... \n");
            runtime->enabled_layers.validation.enabled = true;
            enabled_layers[enabled_layers_count++] = layer->layerName;
        }
    }

    uint32_t extensions_count;
    CHECK_VK(vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, NULL), return false);
    LARRAY(VkExtensionProperties, extensions, extensions_count);
    CHECK_VK(vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, extensions), return false);

    uint32_t enabled_extensions_count = 0;
    LARRAY(const char*, enabled_extensions, extensions_count);

    for (uint32_t i = 0; i < extensions_count; i++) {
        VkExtensionProperties* extension = &extensions[i];

#define X(is_required, prefix, name, _) \
        if (strcmp(extension->extensionName, "VK_"#prefix"_"#name) == 0) { \
            info_print("Enabling instance extension VK_"#prefix"_"#name"\n"); \
            runtime->instance_exts.name.enabled = true; \
            enabled_extensions[enabled_extensions_count++] = extension->extensionName; \
        }
INSTANCE_EXTENSIONS(X)
#undef X
    }

    VkImageCreateFlagBits instance_flags = 0;
    if (runtime->instance_exts.portability_enumeration.enabled)
        instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    CHECK_VK(vkCreateInstance(&(VkInstanceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo) {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pEngineName = "shady",
            .pApplicationName = "business",
            .pNext = NULL,
            .engineVersion = 1,
            .applicationVersion = 1,
            .apiVersion = VK_MAKE_API_VERSION(0, 1, 1, 0)
        },
        .flags = instance_flags,
        .enabledExtensionCount = enabled_extensions_count,
        .ppEnabledExtensionNames = enabled_extensions,
        .enabledLayerCount = enabled_layers_count,
        .ppEnabledLayerNames = enabled_layers,
        .pNext = NULL
    }, NULL, &runtime->instance), return false)

    obtain_instance_pointers(runtime);

    if (runtime->instance_exts.debug_utils.enabled)
        assert(setup_debug_callback(runtime));

    return true;
}


Runtime* initialize_runtime(RuntimeConfig config) {
    Runtime* runtime = malloc(sizeof(Runtime));
    memset(runtime, 0, sizeof(Runtime));
    runtime->config = config;
    runtime->devices = new_list(Device*);
    runtime->programs = new_list(Program*);

    CHECK(initialize_vk_instance(runtime), goto init_fail_free)
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
    for (size_t i = 0; i < entries_count_list(runtime->programs); i++) {
        unload_program(read_list(Program*, runtime->programs)[i]);
    }
    destroy_list(runtime->programs);

    for (size_t i = 0; i < entries_count_list(runtime->devices); i++) {
        shutdown_device(read_list(Device*, runtime->devices)[i]);
    }
    destroy_list(runtime->devices);

    vkDestroyInstance(runtime->instance, NULL);
    free(runtime);
}
