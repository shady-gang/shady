#include "runtime_private.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"

#include <string.h>

static bool fill_available_extensions(VkPhysicalDevice physical_device, size_t* count, const char* enabled_extensions[], bool support_vector[]) {
    if (count)
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
            if (enabled_extensions)
                enabled_extensions[0] = shady_supported_device_extensions_names[i];
            return false;
        }
        if (found && enabled_extensions && count)
            enabled_extensions[(*count)++] = shady_supported_device_extensions_names[i];
    }

    return true;
}

static void figure_out_spirv_version(DeviceCaps* caps) {
    uint32_t major = VK_VERSION_MAJOR(caps->base_properties.apiVersion);
    uint32_t minor = VK_VERSION_MINOR(caps->base_properties.apiVersion);

    assert(major >= 1);
    caps->spirv_version.major = 1;
    if (major == 1 && minor <= 1) {
        // Vulkan 1.1 offers no clear guarantees of supported SPIR-V versions. There is an ext for 1.4 ...
        if (caps->supported_extensions[ShadySupportsKHRspirv_1_4]) {
            caps->spirv_version.minor = 4;
        } else {
            // but there is no way to signal support for spv at or below 1.3, so we just hope 1.3 works out.
            caps->spirv_version.minor = 3;
        }
    } else if (major == 1 && minor == 2) {
        // Vulkan 1.2 guarantees support for spv 1.5
        caps->spirv_version.minor = 5;
    } else {
        // Vulkan 1.3 and later can do spv 1.6
        caps->spirv_version.minor = 6;
    }

    debug_print("Using SPIR-V version %d.%d, on Vulkan %d.%d\n", caps->spirv_version.major, caps->spirv_version.minor, major, minor);
}

static bool fill_basic_device_properties(DeviceCaps* caps) {
    VkPhysicalDeviceProperties2 dp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = NULL
    };
    vkGetPhysicalDeviceProperties2(caps->physical_device, &dp);
    caps->base_properties = dp.properties;

    if (caps->base_properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
        info_print("Rejecting device '%s' because it does not support Vulkan 1.1 or later\n", caps->base_properties.deviceName);
        return false;
    }

    String missing_ext;
    if (!fill_available_extensions(caps->physical_device, NULL, &missing_ext, caps->supported_extensions)) {
        info_print("Rejecting device %s because it lacks support for '%s'\n", caps->base_properties.deviceName, missing_ext);
        return false;
    }

    figure_out_spirv_version(caps);

#ifdef __APPLE__
    // TODO: this is not a proper check
    caps->implementation.is_moltenvk = true;
#endif

    return true;
}

static bool fill_extended_device_properties(DeviceCaps* caps) {
    VkPhysicalDeviceProperties2 dp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = NULL
    };

    caps->extended_properties.subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    append_pnext((VkBaseOutStructure*) &dp, &caps->extended_properties.subgroup);

    if (caps->supported_extensions[ShadySupportsEXTsubgroup_size_control] || caps->base_properties.apiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
        caps->extended_properties.subgroup_size_control.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
        append_pnext((VkBaseOutStructure*) &dp, &caps->extended_properties.subgroup_size_control);
    }

    if (caps->supported_extensions[ShadySupportsEXTexternal_memory_host]) {
        caps->extended_properties.external_memory_host.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        append_pnext((VkBaseOutStructure*) &dp, &caps->extended_properties.external_memory_host);
    }

    vkGetPhysicalDeviceProperties2(caps->physical_device, &dp);

    if (caps->supported_extensions[ShadySupportsEXTsubgroup_size_control] || caps->base_properties.apiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
        caps->subgroup_size.max = caps->extended_properties.subgroup_size_control.maxSubgroupSize;
        caps->subgroup_size.min = caps->extended_properties.subgroup_size_control.minSubgroupSize;
    } else {
        caps->subgroup_size.max = caps->extended_properties.subgroup.subgroupSize;
        caps->subgroup_size.min = caps->extended_properties.subgroup.subgroupSize;
    }
    debug_print("Subgroup size range for device '%s' is [%d; %d]\n", caps->base_properties.deviceName, caps->subgroup_size.min, caps->subgroup_size.max);
    return true;
}

static bool fill_device_features(DeviceCaps* caps) {
    caps->features.base = (VkPhysicalDeviceFeatures2) {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL,
    };

    if (caps->supported_extensions[ShadySupportsKHRshader_subgroup_extended_types] || caps->base_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {
        caps->features.subgroup_extended_types.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR;
        append_pnext((VkBaseOutStructure*) &caps->features.base, &caps->features.subgroup_extended_types);
    }

    if (caps->supported_extensions[ShadySupportsKHRbuffer_device_address] || caps->base_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {
        caps->features.buffer_device_address.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
        append_pnext((VkBaseOutStructure*) &caps->features.base, &caps->features.buffer_device_address);
    }

    if (caps->supported_extensions[ShadySupportsEXTsubgroup_size_control]) {
        caps->features.subgroup_size_control.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        append_pnext((VkBaseOutStructure*) &caps->features.base, &caps->features.subgroup_size_control);
    }

    vkGetPhysicalDeviceFeatures2(caps->physical_device, &caps->features.base);

    return true;
}

static bool fill_queue_properties(DeviceCaps* caps) {
    uint32_t queue_families_count;
    vkGetPhysicalDeviceQueueFamilyProperties2(caps->physical_device, &queue_families_count, NULL);
    LARRAY(VkQueueFamilyProperties2, queue_families_properties, queue_families_count);
    for (size_t i = 0; i < queue_families_count; i++) {
        queue_families_properties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        queue_families_properties[i].pNext = NULL;
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(caps->physical_device, &queue_families_count, queue_families_properties);

    uint32_t compute_queue_family = queue_families_count;
    for (uint32_t i = 0; i < queue_families_count; i++) {
        VkQueueFamilyProperties2 queue_family_properties = queue_families_properties[i];
        if (queue_family_properties.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    if (compute_queue_family >= queue_families_count) {
        info_print("Rejecting device %s because it lacks a compute queue family\n", caps->base_properties.deviceName);
        return false;
    }
    caps->compute_queue_family = compute_queue_family;
    return true;
}

/// Considers a given physical device for running on, returns false if it's unusable, otherwise returns a report in out
static bool get_physical_device_caps(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device, DeviceCaps* out) {
    memset(out, 0, sizeof(DeviceCaps));
    out->physical_device = physical_device;

    if (!fill_basic_device_properties(out))
        goto fail;
    if (!fill_extended_device_properties(out))
        goto fail;
    if (!fill_device_features(out))
        goto fail;
    if (!fill_queue_properties(out))
        goto fail;

    return true;

    fail:
    return false;
}

KeyHash hash_spec_program_key(SpecProgramKey* ptr) {
    return hash_murmur(ptr, sizeof(SpecProgramKey));
}

bool cmp_spec_program_keys(SpecProgramKey* a, SpecProgramKey* b) {
    return memcmp(a, b, sizeof(SpecProgramKey)) == 0;
}

static void obtain_device_pointers(Device* device) {
#define Y(fn_name) ext->fn_name = (PFN_##fn_name) vkGetDeviceProcAddr(device->device, #fn_name);
#define X(_, prefix, name, fns) \
        device->extensions.name.enabled = device->caps.supported_extensions[ShadySupports##prefix##name]; \
        if (device->extensions.name.enabled) { \
            SHADY_UNUSED struct S_##name* ext = &device->extensions.name; \
            fns(Y) \
        }
    DEVICE_EXTENSIONS(X)
#undef Y
#undef X
}

static Device* create_device(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device) {
    Device* device = calloc(1, sizeof(Device));
    device->runtime = runtime;
    CHECK(get_physical_device_caps(runtime, physical_device, &device->caps), assert(false));
    info_print("Initialising device %s\n", device->caps.base_properties.deviceName);

    LARRAY(const char*, enabled_device_exts, ShadySupportedDeviceExtensionsCount);
    size_t enabled_device_exts_count;
    CHECK(fill_available_extensions(physical_device, &enabled_device_exts_count, enabled_device_exts, NULL), assert(false));

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
                .queueFamilyIndex = device->caps.compute_queue_family,
                .pNext = NULL,
            }
        },
        .enabledLayerCount = 0,
        .enabledExtensionCount = enabled_device_exts_count,
        .ppEnabledExtensionNames = enabled_device_exts,
        .pEnabledFeatures = NULL,
        .pNext = &device->caps.features.base,
    }, NULL, &device->device), goto fail;)

    CHECK_VK(vkCreateCommandPool(device->device, &(VkCommandPoolCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .queueFamilyIndex = device->caps.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    }, NULL, &device->cmd_pool), goto delete_device);

    device->specialized_programs = new_dict(SpecProgramKey, SpecProgram*, (HashFn) hash_spec_program_key, (CmpFn) cmp_spec_program_keys);

    vkGetDeviceQueue(device->device, device->caps.compute_queue_family, 0, &device->compute_queue);

    obtain_device_pointers(device);

    return device;

    delete_device:
    vkDestroyDevice(device->device, NULL);

    fail:
    return NULL;
}

bool probe_devices(Runtime* runtime) {
    uint32_t devices_count;
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, NULL), return false)
    LARRAY(VkPhysicalDevice, available_devices, devices_count);
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, available_devices), return false)

    if (devices_count == 0 && !runtime->config.allow_no_devices) {
        error_print("No vulkan devices found!\n");
        error_print("You may be able to diagnose this further using `VK_LOADER_DEBUG=all vulkaninfo`.\n");
        return false;
    }

    for (uint32_t i = 0; i < devices_count; i++) {
        VkPhysicalDevice physical_device = available_devices[i];
        DeviceCaps dummy;
        if (get_physical_device_caps(runtime, physical_device, &dummy)) {
            Device* device = create_device(runtime, physical_device);
            append_list(Device*, runtime->devices, device);
        }
    }

    if (entries_count_list(runtime->devices) == 0 && !runtime->config.allow_no_devices) {
        error_print("No __suitable__ vulkan devices found!\n");
        error_print("This is caused by running on weird hardware configurations. Hardware support might get better in the future.\n");
        return false;
    }

    info_print("Found %d usable devices\n", entries_count_list(runtime->devices));

    return true;
}

void shutdown_device(Device* device) {
    size_t i = 0;
    SpecProgramKey k;
    SpecProgram* sp;
    while (dict_iter(device->specialized_programs, &i, &k, &sp)) {
        destroy_specialized_program(sp);
    }
    destroy_dict(device->specialized_programs);
    vkDestroyCommandPool(device->device, device->cmd_pool, NULL);
    vkDestroyDevice(device->device, NULL);
    free(device);
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

const char* get_device_name(Device* device) { return device->caps.base_properties.deviceName; }
