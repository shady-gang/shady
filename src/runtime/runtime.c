#include "shady/runtime.h"
#include "shady/ir.h"

#include "vulkan/vulkan.h"

#include "log.h"
#include "portability.h"
#include "list.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

#define empty_fns(Y)

#define debug_utils_fns(Y) \
Y(vkCreateDebugUtilsMessengerEXT) \
Y(vkDestroyDebugUtilsMessengerEXT) \

#define INSTANCE_EXTENSIONS(X) \
X(EXT, debug_utils,             debug_utils_fns) \
X(KHR, portability_enumeration, empty_fns)       \

#define DEVICE_EXTENSIONS(X) \
X(EXT, descriptor_indexing,     empty_fns) \
X(KHR, portability_subset,      empty_fns) \

#define E(prefix, name, _) ShadySupports##prefix##name,
enum ShadySupportedInstanceExtensions {
    INSTANCE_EXTENSIONS(E)
    ShadySupportedInstanceExtensionsCount
};
enum ShadySupportedDeviceExtensions {
    DEVICE_EXTENSIONS(E)
    ShadySupportedDeviceExtensionsCount
};
#define S(prefix, name, _) "VK_" #prefix "_" #name,
const char* shady_supported_instance_extensions_names[] = { INSTANCE_EXTENSIONS(S) };
const char* shady_supported_device_extensions_names[] = { DEVICE_EXTENSIONS(S) };
#undef S
#undef E

struct Runtime_ {
    RuntimeConfig config;
    VkInstance instance;
    VkDevice device;

    struct {
        struct {
            bool enabled;
        } validation;
    } enabled_layers;
    struct {
    #define Y(fn_name) PFN_##fn_name fn_name;
    #define X(prefix, name, fns) \
        struct S_##name { \
        bool enabled; \
        fns(Y)  \
        } name;
    INSTANCE_EXTENSIONS(X)
    #undef Y
    #undef X
    } instance_exts;

    VkDebugUtilsMessengerEXT debug_messenger;
};

struct Program_ {
    Runtime* runtime;

    IrArena* arena;
    const Node* program;

    size_t spirv_size;
    char* spirv_bytes;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL the_callback(SHADY_UNUSED VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, SHADY_UNUSED VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, SHADY_UNUSED void* pUserData) {
    warn_print("Validation says: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

static bool setup_debug_callback(Runtime* runtime) {
    CHECK_VK(runtime->instance_exts.debug_utils.vkCreateDebugUtilsMessengerEXT(runtime->instance, &(VkDebugUtilsMessengerCreateInfoEXT) {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT,
        .pfnUserCallback = the_callback,
        .pUserData = NULL
    }, NULL, &runtime->debug_messenger), return false);
    return true;
}

static void obtain_instance_pointers(Runtime* runtime) {
    #define Y(fn_name) ext->fn_name = (PFN_##fn_name) vkGetInstanceProcAddr(runtime->instance, #fn_name);
    #define X(prefix, name, fns) \
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

#define X(prefix, name, _) \
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

static VkDevice initialize_physical_device(SHADY_UNUSED Runtime* runtime, VkPhysicalDevice physical_device) {
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

    VkDevice device;
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
    }, NULL, &device), return NULL)

    return device;
}

static bool initialize_vk_device(Runtime* runtime) {
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
    runtime->device = initialize_physical_device(runtime, physical_device);
    return true;
}

Runtime* initialize_runtime(RuntimeConfig config) {
    Runtime* runtime = malloc(sizeof(Runtime));
    memset(runtime, 0, sizeof(Runtime));
    runtime->config = config;

    if (!initialize_vk_instance(runtime))
        goto init_fail_free;

    if (!initialize_vk_device(runtime))
        goto init_fail;

    info_print("Shady runtime successfully initialized !\n");
    return runtime;

    init_fail:
    vkDestroyInstance(runtime->instance, NULL);

    init_fail_free:
    error_print("Failed to initialise the runtime.\n");
    free(runtime);
    return NULL;
}

void shutdown_runtime(Runtime* runtime) {
    if (!runtime) return;

    vkDestroyDevice(runtime->device, NULL);
    vkDestroyInstance(runtime->instance, NULL);

    free(runtime);
}

static bool compile_program(Program* program, const char* program_src) {
    CompilerConfig config = default_compiler_config();
    config.allow_frontend_syntax = true;
    ArenaConfig arena_config = {};
    program->arena = new_arena(arena_config);
    parse_files(&config, 1, (const char* []){ program_src }, program->arena, &program->program);
    run_compiler_passes(&config, &program->arena, &program->program);
    emit_spirv(&config, program->arena, program->program, &program->spirv_size, &program->spirv_bytes);
    return true;
}

static bool extract_layout(Program* program) {
    CHECK_VK(vkCreatePipelineLayout(program->runtime->device, &(VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pushConstantRangeCount = 0, // TODO !
        .setLayoutCount = 0 // TODO !
    }, NULL, &program->layout), return false);
    return true;
}

static bool create_vk_pipeline(Program* program) {
    CHECK_VK(vkCreateShaderModule(program->runtime->device, &(VkShaderModuleCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = program->spirv_size / 4,
        .pCode = (uint32_t*) program->spirv_bytes
    }, NULL, &program->shader_module), return false);

    CHECK_VK(vkCreateComputePipelines(program->runtime->device, VK_NULL_HANDLE, 1, (VkComputePipelineCreateInfo []) { {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .layout = program->layout,
        .stage = (VkPipelineShaderStageCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .module = program->shader_module,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "the_shader_yo",
            .pSpecializationInfo = NULL
        }
    } }, NULL, &program->pipeline), return false);
    return true;
}

Program* load_program(Runtime* runtime, const char* program_src) {
    Program* program = malloc(sizeof(Program));
    memset(program, 0, sizeof(Program));
    program->runtime = runtime;
    compile_program(program, program_src);
    extract_layout(program);
    create_vk_pipeline(program);
    return program;
}

void launch_kernel(Program* program, int dimx, int dimy, int dimz, int extra_args_count, void** extra_args) {
    error("TODO");
}

