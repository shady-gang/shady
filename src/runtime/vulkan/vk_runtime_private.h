#ifndef SHADY_VK_RUNTIME_PRIVATE_H
#define SHADY_VK_RUNTIME_PRIVATE_H

#include "../runtime_private.h"
#include "shady/ir.h"

#include "portability.h"
#include "arena.h"

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
X(0, EXT_external_memory_host,           external_memory_host_fns) \
X(0, EXT_subgroup_size_control,          empty_fns) \
X(0, KHR_shader_float16_int8,            empty_fns) \
X(0, KHR_8bit_storage,                   empty_fns) \
X(0, KHR_16bit_storage,                  empty_fns) \
X(0, KHR_driver_properties,              empty_fns) \

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

#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

typedef struct VkrSpecProgram_ VkrSpecProgram;

typedef struct VkrBackend_ {
    Backend base;
    VkInstance instance;

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
} VkrBackend;

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
        VkPhysicalDeviceDriverPropertiesKHR driver_properties;
    } properties;
    struct {
        bool is_moltenvk;
    } implementation;
} VkrDeviceCaps;

typedef struct {
    Program* base;
    String entry_point;
} SpecProgramKey;

typedef struct VkrDevice_ VkrDevice;

struct VkrDevice_ {
    Device base;
    VkrBackend* runtime;
    VkrDeviceCaps caps;
    VkDevice device;
    VkCommandPool cmd_pool;
    VkQueue compute_queue;

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

bool probe_vkr_devices(VkrBackend*);

typedef struct VkrBuffer_ {
    Buffer base;
    VkrDevice* device;
    bool imported;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t offset;
    size_t size;
    void* host_ptr;
} VkrBuffer;

VkrBuffer* vkr_allocate_buffer_device(VkrDevice* device, size_t size);
VkrBuffer* vkr_import_buffer_host(VkrDevice* device, void* ptr, size_t size);
bool vkr_can_import_host_memory(VkrDevice* device);

typedef struct VkrCommand_ VkrCommand;

struct VkrCommand_ {
    Command base;
    VkrDevice* device;
    VkCommandBuffer cmd_buf;
    VkFence done_fence;
    bool submitted;
};

VkrCommand* vkr_begin_command(VkrDevice* device);
bool vkr_submit_command(VkrCommand* commands);
void vkr_destroy_command(VkrCommand* commands);
bool vkr_wait_completion(VkrCommand* cmd);

VkrCommand* vkr_launch_kernel(VkrDevice* device, Program* program, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args);

typedef struct {
    size_t num_args;
    const size_t* arg_offset;
    const size_t* arg_size;
    size_t args_size;
} ProgramParamsInfo;

typedef struct ProgramResourceInfo_ ProgramResourceInfo;
struct ProgramResourceInfo_ {
    bool is_bound;
    int set, binding;

    ProgramResourceInfo* parent;
    size_t offset;

    bool host_backed_allocation;
    char* host_ptr;

    char* staging;

    AddressSpace as;
    size_t size;
    VkrBuffer* buffer;

    char* default_data;
};

typedef struct {
    size_t num_resources;
    ProgramResourceInfo** resources;
} ProgramResourcesInfo;

#define MAX_DESCRIPTOR_SETS 4

VkDescriptorType as_to_descriptor_type(AddressSpace as);

struct VkrSpecProgram_ {
    SpecProgramKey key;
    VkrDevice* device;
    Arena* arena;

    Module* specialized_module;

    size_t spirv_size;
    char* spirv_bytes;

    ProgramParamsInfo parameters;
    ProgramResourcesInfo resources;

    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;

    VkDescriptorSetLayout set_layouts[MAX_DESCRIPTOR_SETS];
    size_t required_descriptor_counts_count;
    VkDescriptorPoolSize required_descriptor_counts[16];

    VkDescriptorPool descriptor_pool;
    VkDescriptorSet sets[MAX_DESCRIPTOR_SETS];
};

VkrSpecProgram* get_specialized_program(Program*, String ep, VkrDevice*);
void destroy_specialized_program(VkrSpecProgram*);

static inline void append_pnext(VkBaseOutStructure* s, void* n) {
    while (s->pNext != NULL)
        s = s->pNext;
    s->pNext = n;
    ((VkBaseOutStructure*) n)->pNext = NULL;
}

#endif
