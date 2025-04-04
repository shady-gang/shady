#include "vk_runner_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

static VKAPI_ATTR VkBool32 VKAPI_CALL the_callback(SHADY_UNUSED VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, SHADY_UNUSED VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, SHADY_UNUSED void* pUserData) {
    shd_warn_print("Validation says: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

static bool setup_debug_callback(VkrBackend* runtime) {
    CHECK_VK(runtime->instance_exts.vkCreateDebugUtilsMessengerEXT(runtime->instance, &(VkDebugUtilsMessengerCreateInfoEXT) {
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

static void obtain_instance_pointers(VkrBackend* runtime) {
#define Y(fn_name) \
    runtime->instance_exts.fn_name = (PFN_##fn_name) vkGetInstanceProcAddr(runtime->instance, #fn_name); \
    assert(runtime->instance_exts.fn_name && "loading instance fn pointer "#fn_name" failed");
    INSTANCE_FUNCTIONS(Y)
#undef Y
}

static VkInstance provided_vkinstance = VK_NULL_HANDLE;

void shd_rn_provide_vkinstance(VkInstance instance) {
    provided_vkinstance = instance;
}

static bool initialize_vk_instance(VkrBackend* runtime) {
    uint32_t layers_count;
    CHECK_VK(vkEnumerateInstanceLayerProperties(&layers_count, NULL), return false);
    LARRAY(VkLayerProperties, layer_properties, layers_count);
    CHECK_VK(vkEnumerateInstanceLayerProperties(&layers_count, layer_properties), return false);

    uint32_t enabled_layers_count = 0;
    LARRAY(const char*, enabled_layers, layers_count);

    for (uint32_t i = 0; i < layers_count; i++) {
        VkLayerProperties* layer = &layer_properties[i];

        // Enable validation if the config says so
        if (runtime->base.runner->config.use_validation && strcmp(layer->layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            shd_info_print("Enabling validation... \n");
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

#define X(is_required,  name) \
        if (strcmp(extension->extensionName, "VK_"#name) == 0) { \
            shd_info_print("Enabling instance extension VK_"#name"\n"); \
            runtime->instance_exts.name##_enabled = true; \
            enabled_extensions[enabled_extensions_count++] = extension->extensionName; \
        }
        SHADY_SUPPORTED_INSTANCE_EXTENSIONS(X)
#undef X
    }

    VkImageCreateFlagBits instance_flags = 0;
    if (runtime->instance_exts.KHR_portability_enumeration_enabled)
        instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    VkResult err_create_instance = vkCreateInstance(&(VkInstanceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &(VkApplicationInfo) {
                    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    .pEngineName = "shady",
                    .pApplicationName = "business",
                    .pNext = NULL,
                    .engineVersion = 1,
                    .applicationVersion = 1,
                    .apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0) /* this will still work on VK 1.1+ devices ! */
            },
            .flags = instance_flags,
            .enabledExtensionCount = enabled_extensions_count,
            .ppEnabledExtensionNames = enabled_extensions,
            .enabledLayerCount = enabled_layers_count,
            .ppEnabledLayerNames = enabled_layers,
            .pNext = NULL
    }, NULL, &runtime->instance);
    switch (err_create_instance) {
        case VK_SUCCESS: break;
        case VK_ERROR_INCOMPATIBLE_DRIVER: {
            // Vulkan 1.0 is not worth supporting. It has many API warts and 1.1 fixes many of them.
            // the hardware support is basically identical, so you're not cutting off any devices, just stinky old drivers.
            shd_error_print("vkCreateInstance reported VK_ERROR_INCOMPATIBLE_DRIVER. This most certainly means you're trying to run on a Vulkan 1.0 implementation.\n");
            shd_error_print("This application is written with Vulkan 1.1 as the baseline, you will need to update your Vulkan loader and/or driver.");
            return false;
        }
        default: {
            shd_error_print("vkCreateInstanced failed (%u)\n", err_create_instance);
            return false;
        }
    }

    return true;
}

static void shutdown_vulkan_runtime(VkrBackend* backend) {
    if (!backend) return;

    // don't do that if we were provided the instance !
    if (!provided_vkinstance) {
        if (backend->debug_messenger)
            backend->instance_exts.vkDestroyDebugUtilsMessengerEXT(backend->instance, backend->debug_messenger, NULL);
        vkDestroyInstance(backend->instance, NULL);
    }

    free(backend);
}

Backend* shd_vkr_init(Runner* base) {
    VkrBackend* backend = malloc(sizeof(VkrBackend));
    memset(backend, 0, sizeof(VkrBackend));
    backend->base = (Backend) {
        .runner = base,
        .backend_type = VulkanRuntimeBackend,
        .cleanup = (void(*)()) shutdown_vulkan_runtime,
    };

    if (provided_vkinstance)
        backend->instance = provided_vkinstance;
    else
        CHECK(initialize_vk_instance(backend), goto init_fail_free)
    obtain_instance_pointers(backend);
    if (!provided_vkinstance) {
        if (backend->instance_exts.EXT_debug_utils_enabled)
            assert(setup_debug_callback(backend));

        shd_vkr_probe_devices(backend);
        shd_info_print("Shady Vulkan runner backend successfully initialized, found %d devices.\n", base->devices->elements_count);
    } else {
        shd_info_print("Shady Vulkan runner backend successfully initialized, probing disabled.\n");
    }
    return &backend->base;

    init_fail_free:
    shd_error_print("Failed to initialise the Vulkan back-end.\n");
    free(backend);
    return NULL;
}
