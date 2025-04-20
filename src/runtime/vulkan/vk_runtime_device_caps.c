#include "shady/runtime/vulkan.h"

#include "portability.h"
#include "log.h"

#include <stdlib.h>
#include <string.h>

#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { shd_error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

static inline void append_pnext(VkBaseOutStructure* s, void* n) {
    while (s->pNext != NULL)
        s = s->pNext;
    s->pNext = n;
    ((VkBaseOutStructure*) n)->pNext = NULL;
}

#define S(is_required, name) "VK_" #name,
SHADY_UNUSED static const char* shady_supported_device_extensions_names[] = { SHADY_SUPPORTED_DEVICE_EXTENSIONS(S) };
#undef S

#define R(is_required, _) is_required,
SHADY_UNUSED static const bool shady_is_device_ext_required[] = { SHADY_SUPPORTED_DEVICE_EXTENSIONS(R) };
#undef R

static void figure_out_spirv_version(ShadyVkrPhysicalDeviceCaps* caps) {
    uint32_t major = VK_VERSION_MAJOR(caps->properties.base.properties.apiVersion);
    uint32_t minor = VK_VERSION_MINOR(caps->properties.base.properties.apiVersion);

    assert(major >= 1);
    caps->spirv_version.major = 1;
    if (major == 1 && minor <= 1) {
        // Vulkan 1.1 offers no clear guarantees of supported SPIR-V versions. There is an ext for 1.4 ...
        if (caps->supported_extensions[ShadySupportsKHR_spirv_1_4]) {
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

    shd_debug_print("Using SPIR-V version %d.%d, on Vulkan %d.%d\n", caps->spirv_version.major, caps->spirv_version.minor, major, minor);
}

static bool fill_available_extensions(ShadyVkrPhysicalDeviceCaps* caps) {
    caps->device_extensions_count = 0;

    uint32_t available_count;
    CHECK_VK(vkEnumerateDeviceExtensionProperties(caps->physical_device, NULL, &available_count, NULL), return false);
    LARRAY(VkExtensionProperties, available_exts, available_count);
    CHECK_VK(vkEnumerateDeviceExtensionProperties(caps->physical_device, NULL, &available_count, available_exts), return false);

    for (size_t i = 0; i < SHADY_SUPPORTED_DEVICE_EXTENSIONS_COUNT; i++) {
        bool found = false;
        for (size_t j = 0; j < available_count; j++) {
            if (strcmp(shady_supported_device_extensions_names[i], available_exts[j].extensionName) == 0) {
                found = true;
                break;
            }
        }

        caps->supported_extensions[i] = found;

        if (shady_is_device_ext_required[i] && !found) {
            shd_info_print("Rejecting device %s because it lacks support for '%s'\n", caps->properties.base.properties.deviceName, shady_supported_device_extensions_names[i]);
            return false;
        }
        if (found)
            caps->device_extensions[(caps->device_extensions_count)++] = shady_supported_device_extensions_names[i];
    }

    return true;
}

static bool fill_device_properties(ShadyVkrPhysicalDeviceCaps* caps) {
    caps->properties.base.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(caps->physical_device, &caps->properties.base);

    if (caps->properties.base.properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
        shd_info_print("Rejecting device '%s' because it does not support Vulkan 1.1 or later\n", caps->properties.base.properties.deviceName);
        return false;
    }

    if (!fill_available_extensions(caps)) {
        return false;
    }

    figure_out_spirv_version(caps);

#ifdef __APPLE__
    // TODO: this is not a proper check
    caps->implementation.is_moltenvk = true;
#endif

    caps->properties.subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    append_pnext((VkBaseOutStructure*) &caps->properties.base, &caps->properties.subgroup);

    if (caps->supported_extensions[ShadySupportsEXT_subgroup_size_control] || caps->properties.base.properties.apiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
        caps->properties.subgroup_size_control.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
        append_pnext((VkBaseOutStructure*) &caps->properties.base, &caps->properties.subgroup_size_control);
    }

    if (caps->supported_extensions[ShadySupportsEXT_external_memory_host]) {
        caps->properties.external_memory_host.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        append_pnext((VkBaseOutStructure*) &caps->properties.base, &caps->properties.external_memory_host);
    }

    if (caps->supported_extensions[ShadySupportsKHR_driver_properties]) {
        caps->properties.driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        append_pnext((VkBaseOutStructure*) &caps->properties.base, &caps->properties.driver_properties);
    }

    if (caps->supported_extensions[ShadySupportsKHR_ray_tracing_pipeline]) {
        caps->properties.rt_pipeline_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        append_pnext((VkBaseOutStructure*) &caps->properties.base, &caps->properties.rt_pipeline_properties);
    }

    vkGetPhysicalDeviceProperties2(caps->physical_device, &caps->properties.base);

    if (caps->supported_extensions[ShadySupportsEXT_subgroup_size_control] || caps->properties.base.properties.apiVersion >= VK_MAKE_VERSION(1, 3, 0)) {
        caps->subgroup_size.max = caps->properties.subgroup_size_control.maxSubgroupSize;
        caps->subgroup_size.min = caps->properties.subgroup_size_control.minSubgroupSize;
    } else {
        caps->subgroup_size.max = caps->properties.subgroup.subgroupSize;
        caps->subgroup_size.min = caps->properties.subgroup.subgroupSize;
    }
    shd_debug_print("Subgroup size range for device '%s' is [%d; %d]\n", caps->properties.base.properties.deviceName, caps->subgroup_size.min, caps->subgroup_size.max);
    return true;
}

static void register_ext_feature_impl(size_t* len, VkBaseInStructure** features, size_t* lens, VkBaseInStructure* s, size_t s_len) {
    if (features) {
        features[*len] = s;
    }
    if (lens) {
        lens[*len] = s_len;
    }
    (*len)++;
}

void shd_rt_get_device_caps_ext_features(ShadyVkrPhysicalDeviceCaps* caps, size_t* len, VkBaseInStructure** features, size_t* lens) {
    assert(len);
    *len = 0;

#define register_ext_feature(s) register_ext_feature_impl(len, features, lens, (VkBaseInStructure*) &s, sizeof(s))

    if (caps->supported_extensions[ShadySupportsKHR_shader_subgroup_extended_types] || caps->properties.base.properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {
        caps->features.subgroup_extended_types.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR;
        register_ext_feature(caps->features.subgroup_extended_types);
    }

    if (caps->supported_extensions[ShadySupportsKHR_buffer_device_address] || caps->properties.base.properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {
        caps->features.buffer_device_address.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
        register_ext_feature(caps->features.buffer_device_address);
    }

    if (caps->supported_extensions[ShadySupportsEXT_subgroup_size_control]) {
        caps->features.subgroup_size_control.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        register_ext_feature(caps->features.subgroup_size_control);
    }

    if (caps->supported_extensions[ShadySupportsKHR_shader_float16_int8]) {
        caps->features.float_16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
        register_ext_feature(caps->features.float_16_int8);
    }

    if (caps->supported_extensions[ShadySupportsKHR_8bit_storage]) {
        caps->features.storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
        register_ext_feature(caps->features.storage8);
    }

    if (caps->supported_extensions[ShadySupportsKHR_16bit_storage]) {
        caps->features.storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
        register_ext_feature(caps->features.storage16);
    }

    if (caps->supported_extensions[ShadySupportsKHR_ray_tracing_pipeline]) {
        caps->features.rt_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        register_ext_feature(caps->features.rt_pipeline_features);
    }
}

static bool fill_device_features(ShadyVkrPhysicalDeviceCaps* caps) {
    caps->features.base = (VkPhysicalDeviceFeatures2) {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL,
    };

    LARRAY(VkBaseInStructure*, extended_features, SHADY_SUPPORTED_DEVICE_EXTENSIONS_COUNT);
    size_t len;
    shd_rt_get_device_caps_ext_features(caps, &len, extended_features, NULL);
    for (size_t i = 0; i < len; i++) {
        append_pnext((VkBaseOutStructure*) &caps->features.base, extended_features[i]);
    }

    vkGetPhysicalDeviceFeatures2(caps->physical_device, &caps->features.base);

    if (!caps->features.subgroup_size_control.computeFullSubgroups) {
        shd_warn_print("Potentially broken behaviour on device %s because it does not support computeFullSubgroups", caps->properties.base.properties.deviceName);
        // TODO just outright reject such devices ?
    }

    return true;
}

static bool fill_queue_properties(ShadyVkrPhysicalDeviceCaps* caps) {
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
        bool suitable = queue_family_properties.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT;
        if (suitable) {
            compute_queue_family = i;
            break;
        }
    }
    if (compute_queue_family >= queue_families_count) {
        shd_info_print("Rejecting device %s because it lacks a compute queue family\n", caps->properties.base.properties.deviceName);
        return false;
    }
    caps->compute_queue_family = compute_queue_family;
    return true;
}

/// Considers a given physical device for running on, returns false if it's unusable, otherwise returns a report in out
bool shd_rt_check_physical_device_suitability(VkPhysicalDevice physical_device, ShadyVkrPhysicalDeviceCaps* out) {
    ShadyVkrPhysicalDeviceCaps local_caps;
    ShadyVkrPhysicalDeviceCaps* caps = &local_caps;
    if (out)
        caps = out;

    memset(caps, 0, sizeof(ShadyVkrPhysicalDeviceCaps));
    caps->physical_device = physical_device;

    if (!fill_device_properties(caps))
    goto fail;
    if (!fill_device_features(caps))
    goto fail;
    if (!fill_queue_properties(caps))
    goto fail;

    return true;

    fail:
    return false;
}
