#ifndef SHADY_RUNTIME_PRIVATE_H
#define SHADY_RUNTIME_PRIVATE_H

#include "shady/runtime.h"
#include "shady/ir.h"

#include "vulkan/vulkan.h"

#include <stdbool.h>

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
static const char* shady_supported_instance_extensions_names[] = { INSTANCE_EXTENSIONS(S) };
static const char* shady_supported_device_extensions_names[] = { DEVICE_EXTENSIONS(S) };
#undef S
#undef E

#define CHECK(x, failure_handler) { if (!(x)) { error_print(#x " failed\n"); failure_handler; } }
#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

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

struct Device_ {
    Runtime* runtime;
    VkDevice device;
    VkCommandPool cmd_pool;
    VkQueue compute_queue;

    struct {
    #define Y(fn_name) PFN_##fn_name fn_name;
    #define X(prefix, name, fns) \
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

typedef struct SpecProgram_ {
    Program* base;
    Device* device;

    IrArena* arena;
    const Node* final_program;

    size_t spirv_size;
    char* spirv_bytes;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;
} SpecProgram;

void unload_program(Program*);
void shutdown_device(Device*);

SpecProgram* get_specialized_program(Program*, Device*);

#endif
