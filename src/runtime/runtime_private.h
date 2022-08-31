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

#define INSTANCE_EXTENSIONS(X) \
X(0, EXT, debug_utils,                debug_utils_fns) \
X(0, KHR, portability_enumeration,          empty_fns) \
X(1, KHR, get_physical_device_properties2,  empty_fns) \

#define DEVICE_EXTENSIONS(X) \
X(1, EXT, descriptor_indexing,            empty_fns) \
X(0, KHR, buffer_device_address,          empty_fns) \
X(1, KHR, storage_buffer_storage_class,   empty_fns) \
X(0, KHR, shader_non_semantic_info,       empty_fns) \
X(0, KHR, spirv_1_4,                      empty_fns) \
X(0, KHR, portability_subset,             empty_fns) \
X(0, KHR, shader_subgroup_extended_types, empty_fns) \
X(0, EXT, external_memory,                empty_fns) \
X(0, EXT, external_memory_host,           empty_fns) \

#define E(is_required, prefix, name, _) ShadySupports##prefix##name,
typedef enum {
    INSTANCE_EXTENSIONS(E)
    ShadySupportedInstanceExtensionsCount
} ShadySupportedInstanceExtensions;
typedef enum {
    DEVICE_EXTENSIONS(E)
    ShadySupportedDeviceExtensionsCount
} ShadySupportedDeviceExtensions;
#undef E

#define S(is_required, prefix, name, _) "VK_" #prefix "_" #name,
SHADY_UNUSED static const char* shady_supported_instance_extensions_names[] = { INSTANCE_EXTENSIONS(S) };
SHADY_UNUSED static const char* shady_supported_device_extensions_names[] = { DEVICE_EXTENSIONS(S) };
#undef S

#define R(is_required, _, _1, _2) is_required,
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
    #define X(_, prefix, name, fns) \
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

typedef struct DeviceProperties_ {
    VkPhysicalDevice physical_device;
    uint32_t compute_queue_family;
    struct {
        unsigned int major;
        unsigned int minor;
    } vk_version;
    struct {
        uint8_t major;
        uint8_t minor;
    } spirv_version;
    uint32_t subgroup_size;
    bool supported_extensions[ShadySupportedDeviceExtensionsCount];
    struct {
        bool subgroup_extended_types;
        bool physical_global_ptrs;
    } features;
    struct {
        bool is_moltenvk;
    } implementation;
} DeviceProperties;

struct Device_ {
    Runtime* runtime;
    DeviceProperties properties;
    VkDevice device;
    VkCommandPool cmd_pool;
    VkQueue compute_queue;

    struct {
    #define Y(fn_name) PFN_##fn_name fn_name;
    #define X(_, prefix, name, fns) \
        struct S_##name { \
        bool enabled; \
        fns(Y)  \
        } name;
    DEVICE_EXTENSIONS(X)
    #undef Y
    #undef X
    } extensions;
};

struct Program_ {
    Runtime* runtime;

    IrArena* arena;
    const Node* generic_program;

    struct Dict* specialized;
};

struct SpecProgram_ {
    Program* base;
    Device* device;

    IrArena* arena;
    const Node* final_program;

    size_t spirv_size;
    char* spirv_bytes;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;
};
void unload_program(Program*);
void shutdown_device(Device*);

SpecProgram* get_specialized_program(Program*, Device*);

static inline void append_pnext(VkBaseOutStructure* s, void* n) {
    while (s->pNext != NULL)
        s = s->pNext;
    s->pNext = n;
}

#endif
