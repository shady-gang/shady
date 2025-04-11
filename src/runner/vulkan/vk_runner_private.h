#ifndef SHADY_VK_RUNNER_PRIVATE_H
#define SHADY_VK_RUNNER_PRIVATE_H

#include "../runner_private.h"

#include "shady/runner/vulkan.h"
#include "shady/ir.h"
#include "shady/runtime/vulkan.h"

#include "portability.h"
#include "arena.h"

#include "vulkan/vulkan.h"

#include <stdbool.h>

#define empty_fns(Y)

#define INSTANCE_FUNCTIONS(Y) \
Y(vkCreateDebugUtilsMessengerEXT) \
Y(vkDestroyDebugUtilsMessengerEXT) \

#define DEVICE_FUNCTIONS(Y) \
Y(vkGetMemoryHostPointerPropertiesEXT) \
Y(vkCreateRayTracingPipelinesKHR) \
Y(vkCmdTraceRaysKHR) \
Y(vkGetRayTracingShaderGroupHandlesKHR) \

#define S(is_required, name) "VK_" #name,
SHADY_UNUSED static const char* shady_supported_instance_extensions_names[] = { SHADY_SUPPORTED_INSTANCE_EXTENSIONS(S) };
SHADY_UNUSED static const char* shady_supported_device_extensions_names[] = { SHADY_SUPPORTED_DEVICE_EXTENSIONS(S) };
#undef S

#define R(is_required, _) is_required,
SHADY_UNUSED static const bool shady_is_instance_ext_required[] = { SHADY_SUPPORTED_INSTANCE_EXTENSIONS(R) };
SHADY_UNUSED static const bool shady_is_device_ext_required[] = { SHADY_SUPPORTED_DEVICE_EXTENSIONS(R) };
#undef R

#define CHECK_VK(x, failure_handler) { VkResult the_result_ = x; if (the_result_ != VK_SUCCESS) { shd_error_print(#x " failed (code %d)\n", the_result_); failure_handler; } }

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
    #define X(_, name)  bool name##_enabled;
        SHADY_SUPPORTED_INSTANCE_EXTENSIONS(X)
    #undef X
    #define Y(fn_name) PFN_##fn_name fn_name;
        INSTANCE_FUNCTIONS(Y)
    #undef F
    } instance_exts;

    VkDebugUtilsMessengerEXT debug_messenger;
} VkrBackend;

typedef struct {
    Program* base;
    String entry_point;
    ExecutionModel em;
} SpecProgramKey;

typedef struct VkrDevice_ VkrDevice;

struct VkrDevice_ {
    Device base;
    VkrBackend* runtime;
    ShadyVkrPhysicalDeviceCaps caps;
    VkDevice device;
    bool owns_vkdevice;
    VkCommandPool cmd_pool;
    VkQueue compute_queue;

    struct {
    #define X(_, name) bool name##_enabled;
    SHADY_SUPPORTED_DEVICE_EXTENSIONS(X)
#define Y(fn_name) PFN_##fn_name fn_name;
        DEVICE_FUNCTIONS(Y)
#undef F
    #undef X
    } extensions;

    struct Dict* specialized_programs;
};

bool shd_vkr_probe_devices(VkrBackend* runtime);

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

VkrBuffer* shd_vkr_allocate_buffer_device(VkrDevice* device, size_t size);
VkrBuffer* shd_vkr_import_buffer_host(VkrDevice* device, void* ptr, size_t size);
bool shd_vkr_can_import_host_memory(VkrDevice* device);
void shd_vkr_destroy_buffer(VkrBuffer* buffer);

typedef struct VkrCommand_ VkrCommand;

typedef struct {
    RuntimeInterfaceItem interface_item;

    VkrBuffer* scratch;
} VkrDispatchInterfaceItem;

struct VkrCommand_ {
    Command base;
    VkrDevice* device;
    VkCommandBuffer cmd_buf;
    VkFence done_fence;
    bool submitted;

    VkrSpecProgram* launched_program;
    VkrDispatchInterfaceItem* launch_interface_items;

    uint64_t* profiled_gpu_time;
    VkQueryPool query_pool;
};

VkrCommand* shd_vkr_begin_command(VkrDevice* device);
bool shd_vkr_submit_command(VkrCommand* cmd);
void shd_vkr_destroy_command(VkrCommand* cmd);
bool shd_vkr_wait_completion(VkrCommand* cmd);

VkrCommand* shd_vkr_launch_kernel(VkrDevice* device, Program* program, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions* options);

typedef struct {
    RuntimeInterfaceItem interface_item;

    void* host_owning_ptr;
    VkrBuffer* buffer;
    size_t per_invocation_size;
} VkrProgramInterfaceItem;

void shd_vkr_populate_interface(VkrSpecProgram* spec);
size_t shd_vkr_get_push_constant_size(VkrSpecProgram* program);

#define MAX_DESCRIPTOR_SETS 4

struct VkrSpecProgram_ {
    SpecProgramKey key;
    VkrDevice* device;
    Arena* arena;

    CompilerConfig specialized_config;
    TargetConfig specialized_target;
    Module* specialized_module;

    size_t spirv_size;
    char* spirv_bytes;

    size_t interface_items_count;
    VkrProgramInterfaceItem* interface_items;

    VkShaderStageFlagBits stage;

    struct {
        VkStridedDeviceAddressRegionKHR rg_sbt;
        VkrBuffer* rg_sbt_buffer;
    } rt;

    VkPipelineBindPoint bind_point;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkShaderModule shader_module;

    VkDescriptorSetLayout set_layouts[MAX_DESCRIPTOR_SETS];
    size_t required_descriptor_counts_count;
    VkDescriptorPoolSize required_descriptor_counts[16];

    VkDescriptorPool descriptor_pool;
    VkDescriptorSet sets[MAX_DESCRIPTOR_SETS];
};

TargetConfig shd_vkr_get_device_target_config(VkrDevice* device);

VkrSpecProgram* shd_vkr_get_specialized_program(Program* program, String entry_point, VkrDevice* device);
void shd_vkr_destroy_specialized_program(VkrSpecProgram* spec);

static inline void append_pnext(VkBaseOutStructure* s, void* n) {
    while (s->pNext != NULL)
        s = s->pNext;
    s->pNext = n;
    ((VkBaseOutStructure*) n)->pNext = NULL;
}

#endif
