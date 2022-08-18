#include "runtime_private.h"

#include "log.h"
#include "portability.h"

static VkPhysicalDevice pick_device(SHADY_UNUSED Runtime* runtime, uint32_t devices_count, VkPhysicalDevice available_devices[]) {
    for (uint32_t i = 0; i < devices_count; i++) {
        VkPhysicalDevice physical_device = available_devices[i];
        VkPhysicalDeviceProperties device_properties;
        vkGetPhysicalDeviceProperties(physical_device, &device_properties);

        if (device_properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0))
            continue;

        return available_devices[i];
    }

    return NULL;
}

static Device* create_device(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device) {
    uint32_t queue_families_count;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &queue_families_count, NULL);
    LARRAY(VkQueueFamilyProperties2, queue_families_properties, queue_families_count);
    for (size_t i = 0; i < queue_families_count; i++) {
        queue_families_properties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        queue_families_properties[i].pNext = NULL;
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &queue_families_count, queue_families_properties);

    uint32_t compute_queue_family = -1;
    for (uint32_t i = 0; i < queue_families_count; i++) {
        VkQueueFamilyProperties2 queue_family_properties = queue_families_properties[i];
        if (queue_family_properties.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }

    if (compute_queue_family == -1)
        return NULL;

    Device* device = calloc(1, sizeof(Device));
    device->runtime = runtime;

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
                .queueFamilyIndex = compute_queue_family,
                .pNext = NULL,
            }
        },
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
        .pNext = NULL,
    }, NULL, &device->device), return NULL)

    CHECK_VK(vkCreateCommandPool(device->device, &(VkCommandPoolCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .queueFamilyIndex = compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    }, NULL, &device->cmd_pool), goto delete_device);

    vkGetDeviceQueue(device->device, compute_queue_family, 0, &device->compute_queue);

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
