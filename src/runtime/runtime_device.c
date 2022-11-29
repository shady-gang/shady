#include "runtime_private.h"

#include "log.h"
#include "portability.h"

#include <string.h>

static bool fill_available_extensions(VkPhysicalDevice physical_device, size_t* count, const char* enabled_extensions[], bool support_vector[]) {
    *count = 0;

    uint32_t available_count;
    CHECK_VK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &available_count, NULL), return false);
    LARRAY(VkExtensionProperties, available_exts, available_count);
    CHECK_VK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &available_count, available_exts), return false);

    for (size_t i = 0; i < ShadySupportedDeviceExtensionsCount; i++) {
        bool found = false;
        for (size_t j = 0; j < available_count; j++) {
            if (strcmp(shady_supported_device_extensions_names[i], available_exts[j].extensionName) == 0) {
                found = true;
                break;
            }
        }

        if (support_vector)
            support_vector[i] = found;

        if (is_device_ext_required[i] && !found) {
            enabled_extensions[0] = shady_supported_device_extensions_names[i];
            return false;
        }
        if (found)
            enabled_extensions[(*count)++] = shady_supported_device_extensions_names[i];
    }

    return true;
}

static void figure_out_spirv_version(DeviceProperties* device) {
    assert(device->vk_version.major >= 1);
    device->spirv_version.major = 1;
    if (device->vk_version.major == 1 && device->vk_version.minor <= 1) {
        // Vulkan 1.1 offers no clear guarantees of supported SPIR-V versions. There is an ext for 1.4 ...
        if (device->supported_extensions[ShadySupportsKHRspirv_1_4]) {
            device->spirv_version.minor = 4;
        } else {
            // but there is no way to signal support for spv at or below 1.3, so we just hope 1.3 works out.
            device->spirv_version.minor = 3;
        }
    } else if (device->vk_version.major == 1 && device->vk_version.minor == 2) {
        // Vulkan 1.2 guarantees support for spv 1.5
        device->spirv_version.minor = 5;
    } else {
        // Vulkan 1.3 and later can do spv 1.6
        device->spirv_version.minor = 6;
    }

    debug_print("Using SPIR-V version %d.%d, on Vulkan %d.%d\n", device->spirv_version.major, device->spirv_version.minor, device->vk_version.major, device->vk_version.minor);
}

/// Considers a given physical device for running on, returns false if it's unusable, otherwise returns a report in out
static bool get_physical_device_properties(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device, DeviceProperties* out) {
    memset(out, 0, sizeof(DeviceProperties));
    out->physical_device = physical_device;

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        .pNext = NULL
    };
    VkPhysicalDeviceProperties2 dp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = NULL
    };
    append_pnext((VkBaseOutStructure*) &dp, &subgroup_properties);
    vkGetPhysicalDeviceProperties2(physical_device, &dp);

    if (dp.properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
        info_print("Rejecting device '%s' because it does not support Vulkan 1.1 or later\n", dp.properties.deviceName);
        return false;
    }

    LARRAY(const char*, exts, ShadySupportedDeviceExtensionsCount);
    size_t c;
    if (!fill_available_extensions(physical_device, &c, exts, out->supported_extensions)) {
        info_print("Rejecting device %s because it lacks support for '%s'\n", dp.properties.deviceName, exts[0]);
        return false;
    }

    out->vk_version.major = VK_VERSION_MAJOR(dp.properties.apiVersion);
    out->vk_version.minor = VK_VERSION_MINOR(dp.properties.apiVersion);
    figure_out_spirv_version(out);

#ifdef __APPLE__
    // TODO: this is not a proper check
    out->implementation.is_moltenvk = true;
#endif

    VkPhysicalDeviceFeatures2 df = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL
    };

    VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR subgroup_extended_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
        .pNext = NULL,
    };
    if (out->supported_extensions[ShadySupportsKHRshader_subgroup_extended_types] || dp.properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0))
        append_pnext((VkBaseOutStructure*) &df, &subgroup_extended_features);

    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bda_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
        .pNext = NULL,
    };

    if (out->supported_extensions[ShadySupportsEXTbuffer_device_address] || dp.properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0))
        append_pnext((VkBaseOutStructure*) &df, &bda_features);

    vkGetPhysicalDeviceFeatures2(physical_device, &df);

    if (!df.features.shaderInt64) {
        info_print("Rejecting device '%s' because it does not support 64-bit integers in shaders\n", dp.properties.deviceName);
        return false;
    }

    out->features.subgroup_extended_types = subgroup_extended_features.shaderSubgroupExtendedTypes;
    if (!bda_features.bufferDeviceAddress) {
        info_print("Rejecting device '%s' because it does not buffer device addresses\n", dp.properties.deviceName);
        return false;
    }
    out->features.physical_global_ptrs = bda_features.bufferDeviceAddress;

    out->subgroup_size = subgroup_properties.subgroupSize;
    info_print("Subgroup size for device '%s' is %d\n", dp.properties.deviceName, out->subgroup_size);

    uint32_t queue_families_count;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &queue_families_count, NULL);
    LARRAY(VkQueueFamilyProperties2, queue_families_properties, queue_families_count);
    for (size_t i = 0; i < queue_families_count; i++) {
        queue_families_properties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        queue_families_properties[i].pNext = NULL;
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &queue_families_count, queue_families_properties);

    uint32_t compute_queue_family = queue_families_count;
    for (uint32_t i = 0; i < queue_families_count; i++) {
        VkQueueFamilyProperties2 queue_family_properties = queue_families_properties[i];
        if (queue_family_properties.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    if (compute_queue_family >= queue_families_count) {
        info_print("Rejecting device %s because it lacks a compute queue family\n", dp.properties.deviceName);
        return NULL;
    }
    out->compute_queue_family = compute_queue_family;

    return true;
}

static VkPhysicalDevice pick_device(Runtime* runtime, uint32_t devices_count, VkPhysicalDevice available_devices[]) {
    for (uint32_t i = 0; i < devices_count; i++) {
        VkPhysicalDevice physical_device = available_devices[i];
        DeviceProperties dummy;
        if (get_physical_device_properties(runtime, physical_device, &dummy))
            return available_devices[i];
    }

    return NULL;
}

static Device* create_device(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device) {
    Device* device = calloc(1, sizeof(Device));
    device->runtime = runtime;
    CHECK(get_physical_device_properties(runtime, physical_device, &device->properties), assert(false));

    LARRAY(const char*, enabled_device_exts, ShadySupportedDeviceExtensionsCount);
    size_t enabled_device_exts_count;
    CHECK(fill_available_extensions(physical_device, &enabled_device_exts_count, enabled_device_exts, NULL), assert(false));

    VkPhysicalDeviceFeatures2 enabled_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL,
        .features = {
            .shaderInt64 = true
        }
    };
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR subgroup_extended_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
        .pNext = NULL,
        .shaderSubgroupExtendedTypes = true
    };
    if (device->properties.features.subgroup_extended_types)
        append_pnext((VkBaseOutStructure*) &enabled_features, &subgroup_extended_features);

    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bda_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
        .pNext = NULL,
        .bufferDeviceAddress = true,
    };
    if (device->properties.features.physical_global_ptrs)
        append_pnext((VkBaseOutStructure*) &enabled_features, &bda_features);

    CHECK_VK(vkCreateDevice(physical_device, &(VkDeviceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = (const VkDeviceQueueCreateInfo []) {
            {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .flags = 0,
                .pQueuePriorities = (const float []) { 1.0f },
                .queueCount = 1,
                .queueFamilyIndex = device->properties.compute_queue_family,
                .pNext = NULL,
            }
        },
        .enabledLayerCount = 0,
        .enabledExtensionCount = enabled_device_exts_count,
        .ppEnabledExtensionNames = enabled_device_exts,
        .pEnabledFeatures = NULL,
        .pNext = &enabled_features,
    }, NULL, &device->device), return NULL)

    CHECK_VK(vkCreateCommandPool(device->device, &(VkCommandPoolCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .queueFamilyIndex = device->properties.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    }, NULL, &device->cmd_pool), goto delete_device);

    vkGetDeviceQueue(device->device, device->properties.compute_queue_family, 0, &device->compute_queue);
    return device;

    delete_device:
    vkDestroyDevice(device->device, NULL);
    return NULL;
}

Device* initialize_device(Runtime* runtime) {
    uint32_t devices_count;
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, NULL), return false)
    LARRAY(VkPhysicalDevice, devices, devices_count);
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, devices), return false)

    if (devices_count == 0) {
        error_print("No vulkan devices found!\n");
        error_print("You may be able to diagnose this further using `VK_LOADER_DEBUG=all vulkaninfo`.\n");
        return false;
    }

    VkPhysicalDevice physical_device = pick_device(runtime, devices_count, devices);
    if (physical_device == NULL) {
        error_print("No __suitable__ vulkan devices found!\n");
        error_print("This is caused by running on weird hardware configurations. Hardware support might get better in the future.\n");
        return false;
    }

    return create_device(runtime, physical_device);
}

void shutdown_device(Device* device) {
    free(device);
}
