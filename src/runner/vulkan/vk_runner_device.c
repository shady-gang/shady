#include "vk_runner_private.h"
#include "shady/driver.h"

#include "log.h"
#include "portability.h"
#include "list.h"
#include "dict.h"

#include <string.h>

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

static KeyHash hash_spec_program_key(SpecProgramKey* ptr) {
    return shd_hash(ptr->base, sizeof(Program*)) ^ shd_hash_string(&ptr->entry_point) ^ shd_hash(&ptr->em, sizeof(ptr->em));
}

static bool cmp_spec_program_keys(SpecProgramKey* a, SpecProgramKey* b) {
	assert(!!a & !!b);
    return a->base == b->base && strcmp(a->entry_point, b->entry_point) == 0 && a->em == b->em;
}

static void obtain_device_pointers(VkrDevice* device) {
#define Y(fn_name) device->extensions.fn_name = (PFN_##fn_name) vkGetDeviceProcAddr(device->device, #fn_name);
                   //assert(device->extensions.fn_name && "loading device fn pointer "#fn_name" failed");
    DEVICE_FUNCTIONS(Y)
#undef X
}

static bool create_vk_device(VkrDevice* device) {
    CHECK_VK(vkCreateDevice(device->caps.physical_device, &(VkDeviceCreateInfo) {
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
        .enabledExtensionCount = device->caps.device_extensions_count,
        .ppEnabledExtensionNames = device->caps.device_extensions,
        .pEnabledFeatures = NULL,
        .pNext = &device->caps.features.base,
    }, NULL, &device->device), return false;)
    return true;
}

static void shutdown_vkr_device(VkrDevice* device) {
    size_t i = 0;
    SpecProgramKey k;
    VkrSpecProgram* sp;
    while (shd_dict_iter(device->specialized_programs, &i, &k, &sp)) {
        shd_vkr_destroy_specialized_program(sp);
    }
    shd_destroy_dict(device->specialized_programs);
    vkDestroyCommandPool(device->device, device->cmd_pool, NULL);
    if (device->owns_vkdevice)
        vkDestroyDevice(device->device, NULL);
    free(device);
}

static const char* get_vkr_device_name(VkrDevice* device) { return device->caps.properties.base.properties.deviceName; }

TargetConfig shd_vkr_get_device_target_config(const CompilerConfig* compiler_config, VkrDevice* device) {
    TargetConfig target_config = shd_default_target_config();
    shd_driver_configure_defaults_for_target(&target_config, compiler_config, TgtSPV);
    target_config.subgroup_size = device->caps.subgroup_size.max;
#ifdef VK_KHR_shader_maximal_reconvergence
    target_config.capabilities.maximal_reconvergence = device->caps.features.maximal_reconvergence_features.shaderMaximalReconvergence;
    if (!target_config.capabilities.maximal_reconvergence)
        shd_log_fmt(WARN, "Maximal reconvergence is not supported on this device.\n");
#else
    target_config.capabilities.maximal_reconvergence = false;
    shd_log_fmt(WARN, "Maximal reconvergence is not supported in this build.\n");
#endif
    return target_config;
}

static VkrDevice* create_vkr_device(VkrBackend* runtime, ShadyVkrPhysicalDeviceCaps caps, VkDevice vk_device) {
    VkrDevice* device = calloc(1, sizeof(VkrDevice));
    device->base = (Device) {
        .backend = VulkanRuntimeBackend,
        .cleanup = (void(*)(Device*)) shutdown_vkr_device,
        .get_name = (String(*)(Device*)) get_vkr_device_name,
        .allocate_buffer = (Buffer* (*)(Device*, size_t)) shd_vkr_allocate_buffer_device,
        .import_host_memory_as_buffer = (Buffer* (*)(Device*, void*, size_t)) shd_vkr_import_buffer_host,
        .launch_kernel = (Command* (*)(Device*, Program*, String, int, int, int, int, void**, ExtraKernelOptions*)) shd_vkr_launch_kernel,
        .can_import_host_memory = (bool (*)(Device*)) shd_vkr_can_import_host_memory,
        .get_device_target_config = (TargetConfig (*)(const CompilerConfig*, Device*)) shd_vkr_get_device_target_config,
    };
    device->runtime = runtime;
    device->caps = caps;
    shd_info_print("Initialising device %s\n", device->caps.properties.base.properties.deviceName);

    if (vk_device)
        device->device = vk_device;
    else
        CHECK(create_vk_device(device), goto fail);

    CHECK_VK(vkCreateCommandPool(device->device, &(VkCommandPoolCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .queueFamilyIndex = device->caps.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    }, NULL, &device->cmd_pool), goto delete_device);

    device->specialized_programs = shd_new_dict(SpecProgramKey, VkrSpecProgram*, (HashFn) hash_spec_program_key, (CmpFn) cmp_spec_program_keys);

    vkGetDeviceQueue(device->device, device->caps.compute_queue_family, 0, &device->compute_queue);

    obtain_device_pointers(device);

    return device;

    delete_device:
    vkDestroyDevice(device->device, NULL);

    fail:
    return NULL;
}

static VkrBackend* open_backend(Runner* runner) {
    for (size_t i = 0; i < shd_list_count(runner->backends); i++) {
        Backend* be = shd_read_list(Backend*, runner->backends)[i];
        if (be->backend_type == VulkanRuntimeBackend)
            return (VkrBackend*) be;
    }
    shd_error("Failed to find the Vulkan backend (is it enabled?)");
}

Device* shd_rn_open_vkdevice_with_caps(Runner* runner, ShadyVkrPhysicalDeviceCaps caps, VkDevice vk_device) {
    assert(vk_device);
    VkrBackend* backend = open_backend(runner);
    VkrDevice* device = create_vkr_device(backend, caps, vk_device);
    shd_list_append(Device*, backend->base.runner->devices, device);
    return (Device*) device;
}

Device* shd_rn_open_vkdevice(Runner* runner, VkPhysicalDevice physical_device, VkDevice vk_device) {
    ShadyVkrPhysicalDeviceCaps caps;
    if (!shd_rt_check_physical_device_suitability(physical_device, &caps))
        return NULL;
    return shd_rn_open_vkdevice_with_caps(runner, caps, vk_device);
}

bool shd_vkr_probe_devices(VkrBackend* runtime) {
    uint32_t devices_count;
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, NULL), return false)
    LARRAY(VkPhysicalDevice, available_devices, devices_count);
    CHECK_VK(vkEnumeratePhysicalDevices(runtime->instance, &devices_count, available_devices), return false)

    if (devices_count == 0 && !runtime->base.runner->config.allow_no_devices) {
        shd_error_print("No vulkan devices found!\n");
        shd_error_print("You may be able to diagnose this further using `VK_LOADER_DEBUG=all vulkaninfo`.\n");
        return false;
    }

    for (uint32_t i = 0; i < devices_count; i++) {
        VkPhysicalDevice physical_device = available_devices[i];
        ShadyVkrPhysicalDeviceCaps caps;
        if (shd_rt_check_physical_device_suitability(physical_device, &caps)) {
            VkrDevice* device = create_vkr_device(runtime, caps, VK_NULL_HANDLE);
            device->owns_vkdevice = true;
            shd_list_append(Device*, runtime->base.runner->devices, device);
        }
    }

    if (shd_list_count(runtime->base.runner->devices) == 0 && !runtime->base.runner->config.allow_no_devices) {
        shd_error_print("No __suitable__ vulkan devices found!\n");
        shd_error_print("This is caused by running on weird hardware configurations. Hardware support might get better in the future.\n");
        return false;
    }

    shd_info_print("Found %d usable devices\n", shd_list_count(runtime->base.runner->devices));

    return true;
}

VkDevice shd_rn_get_vkdevice(Device* device) {
    assert(device->backend == VulkanRuntimeBackend);
    VkrDevice* vkr_device = (VkrDevice*) device;
    return vkr_device->device;
}
