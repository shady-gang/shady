#ifndef SHADY_RUNTIME_PRIVATE_H
#define SHADY_RUNTIME_PRIVATE_H

#include "shady/runtime.h"
#include "shady/ir.h"

#include "portability.h"

#include "vulkan/vulkan.h"

#include <stdbool.h>

#define empty_fns(Y)

#define debug_utils_fns(Y) \
Y(vkCreateDebugUtilsMessengerEXT) \
Y(vkDestroyDebugUtilsMessengerEXT) \

#define external_memory_host_fns(Y) \
Y(vkGetMemoryHostPointerPropertiesEXT) \

#define INSTANCE_EXTENSIONS(X) \
X(0, EXT_debug_utils,                debug_utils_fns) \
X(0, KHR_portability_enumeration,          empty_fns) \
X(1, KHR_get_physical_device_properties2,  empty_fns) \

#define DEVICE_EXTENSIONS(X) \
X(0, EXT_descriptor_indexing,            empty_fns) \
X(1, KHR_buffer_device_address,          empty_fns) \
X(1, KHR_storage_buffer_storage_class,   empty_fns) \
X(0, KHR_shader_non_semantic_info,       empty_fns) \
X(0, KHR_spirv_1_4,                      empty_fns) \
X(0, KHR_portability_subset,             empty_fns) \
X(0, KHR_shader_subgroup_extended_types, empty_fns) \
X(0, EXT_external_memory,                empty_fns) \
X(1, EXT_external_memory_host,           external_memory_host_fns) \
X(0, EXT_subgroup_size_control,          empty_fns) \
X(0, KHR_shader_float16_int8,            empty_fns) \
X(0, KHR_8bit_storage,                   empty_fns) \
X(0, KHR_16bit_storage,                  empty_fns) \

#define E(is_required, name, _) ShadySupports##name,
typedef enum {
    INSTANCE_EXTENSIONS(E)
    ShadySupportedInstanceExtensionsCount
} ShadySupportedInstanceExtensions;
typedef enum {
    DEVICE_EXTENSIONS(E)
    ShadySupportedDeviceExtensionsCount
} ShadySupportedDeviceExtensions;
#undef E

#define S(is_required, name, _) "VK_" #name,
SHADY_UNUSED static const char* shady_supported_instance_extensions_names[] = { INSTANCE_EXTENSIONS(S) };
SHADY_UNUSED static const char* shady_supported_device_extensions_names[] = { DEVICE_EXTENSIONS(S) };
#undef S

#define R(is_required, _, _2) is_required,
SHADY_UNUSED static const bool is_instance_ext_required[] = { INSTANCE_EXTENSIONS(R) };
SHADY_UNUSED static const bool is_device_ext_required[] = { DEVICE_EXTENSIONS(R) };
#undef R

#define CHECK(x, failure_handler) { if (!(x)) { error_print(#x " failed\n"); failure_handler; } }
#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

typedef struct SpecProgram_ SpecProgram;

struct Runtime_ {
    RuntimeConfig config;
    VkInstance instance;
    struct List* devices;
    struct List* programs;

    struct {
        struct {
            bool enabled;
        } validation;
    } enabled_layers;

    struct {
    #define Y(fn_name) PFN_##fn_name fn_name;
    #define X(_, name, fns) \
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

typedef struct {
    VkPhysicalDevice physical_device;

    bool supported_extensions[ShadySupportedDeviceExtensionsCount];

    uint32_t compute_queue_family;

    struct {
        uint8_t major;
        uint8_t minor;
    } spirv_version;
    struct {
        uint32_t min, max;
    } subgroup_size;
    struct {
        VkPhysicalDeviceFeatures2 base;
        VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR subgroup_extended_types;
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR buffer_device_address;
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control;
        VkPhysicalDeviceShaderFloat16Int8Features float_16_int8;
        VkPhysicalDevice8BitStorageFeatures storage8;
        VkPhysicalDevice16BitStorageFeatures storage16;
    } features;
    struct {
        VkPhysicalDeviceProperties2 base;
        VkPhysicalDeviceSubgroupProperties subgroup;
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control;
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_host;
    } properties;
    struct {
        bool is_moltenvk;
    } implementation;
} DeviceCaps;

typedef struct StagingBuffer_ {
    void* ptr;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
} StagingBuffer;

bool resize_staging_buffer(Device* device, size_t size);
void free_staging_buffer(Device* device);

struct Device_ {
    Runtime* runtime;
    DeviceCaps caps;
    VkDevice device;
    VkCommandPool cmd_pool;
    VkQueue compute_queue;

    StagingBuffer staging_buffer;

    struct {
    #define Y(fn_name) PFN_##fn_name fn_name;
    #define X(_, name, fns) \
        struct S_##name { \
        bool enabled; \
        fns(Y)  \
        } name;
    DEVICE_EXTENSIONS(X)
    #undef Y
    #undef X
    } extensions;

    struct Dict* specialized_programs;
};

bool probe_devices(Runtime* runtime);

struct Command_ {
    Device* device;
    VkCommandBuffer cmd_buf;
    VkFence done_fence;
    bool submitted;
};

Command* begin_command(Device* device);
bool submit_command(Command* commands);
void destroy_command(Command* commands);

struct Program_ {
    Runtime* runtime;

    IrArena* arena;
    Module* generic_program;
};

typedef struct EntryPointInfo_ {
    size_t num_args;
    const size_t* arg_offset;
    const size_t* arg_size;
    size_t args_size;
} EntryPointInfo;

struct SpecProgram_ {
    Program* base;
    Device* device;

    Module* module;

    size_t spirv_size;
    char* spirv_bytes;

    EntryPointInfo entrypoint;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;
};
void unload_program(Program*);
void shutdown_device(Device*);

SpecProgram* get_specialized_program(Program*, Device*);
void destroy_specialized_program(SpecProgram*);

static inline void append_pnext(VkBaseOutStructure* s, void* n) {
    while (s->pNext != NULL)
        s = s->pNext;
    s->pNext = n;
    ((VkBaseOutStructure*) n)->pNext = NULL;
}

#endif
