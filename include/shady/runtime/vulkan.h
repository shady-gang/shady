#ifndef SHD_RUNTIME_VULKAN
#define SHD_RUNTIME_VULKAN

#include "shady/ir/base.h"
#include "vulkan/vulkan.h"

typedef struct {
    enum {
        SHD_RII_Dst_PushConstant,
        SHD_RII_Dst_Descriptor,
    } dst_kind;
    union {
        struct {
            size_t offset, size;
            size_t param_idx;
        } push_constant;
        struct {
            uint32_t set, binding;
            VkDescriptorType type;
        } descriptor;
    } dst_details;
    enum {
        SHD_RII_Src_Param,
        SHD_RII_Src_LiftedConstant,
        SHD_RII_Src_ScratchBuffer
    } src_kind;
    union {
        struct {
            size_t param_idx;
        } param;
        struct {
            const Node* constant;
        } lifted_constant;
        struct {
            const Node* per_invocation_size;
        } scratch_buffer;
    } src_details;
} RuntimeInterfaceItem;

void shd_vkr_get_runtime_dependencies(Module*, size_t* count, RuntimeInterfaceItem* out);

typedef struct {
    VkDescriptorSetLayoutCreateInfo set_layout;
} ShdDescriptorSetLayout;

void shd_vkr_get_descriptor_layouts(Module*, size_t* count, ShdDescriptorSetLayout** out);
void shd_vkr_free_descriptor_set_layouts();

void shd_vkr_write_push_constants(Module*, size_t arguments_count, void** arguments);

#define SHADY_SUPPORTED_INSTANCE_EXTENSIONS(X) \
X(0, EXT_debug_utils) \
X(0, KHR_portability_enumeration) \
X(1, KHR_get_physical_device_properties2) \

#define SHADY_SUPPORTED_DEVICE_EXTENSIONS(X) \
X(0, EXT_descriptor_indexing) \
X(1, KHR_buffer_device_address) \
X(1, KHR_storage_buffer_storage_class) \
X(0, KHR_shader_non_semantic_info) \
X(0, KHR_spirv_1_4) \
X(0, KHR_portability_subset) \
X(0, KHR_shader_subgroup_extended_types) \
X(0, EXT_external_memory) \
X(0, EXT_external_memory_host) \
X(0, EXT_subgroup_size_control) \
X(0, KHR_shader_float16_int8) \
X(0, KHR_8bit_storage) \
X(0, KHR_16bit_storage) \
X(0, KHR_driver_properties) \
X(0, KHR_ray_tracing_pipeline) \

#define E(is_required, name) ShadySupports##name,
typedef enum {
    SHADY_SUPPORTED_INSTANCE_EXTENSIONS(E)
    SHADY_SUPPORTED_INSTANCE_EXTENSIONS_COUNT
} ShadySupportedInstanceExtensions;

typedef enum {
    SHADY_SUPPORTED_DEVICE_EXTENSIONS(E)
    SHADY_SUPPORTED_DEVICE_EXTENSIONS_COUNT
} ShadySupportedDeviceExtensions;
#undef E

/// This structs describes what a given physical device can do, and is used by the JIT compiler to fine-tune a module to a particular device
typedef struct {
    VkPhysicalDevice physical_device;

    // list of supported and enabled extensions
    // by default, shd_rt_check_physical_device_suitability will enable all optional supported extensions
    size_t device_extensions_count;
    const char* device_extensions[SHADY_SUPPORTED_DEVICE_EXTENSIONS_COUNT];

    // same thing but as a boolean vector
    bool supported_extensions[SHADY_SUPPORTED_DEVICE_EXTENSIONS_COUNT];

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
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features;
    } features;
    struct {
        VkPhysicalDeviceProperties2 base;
        VkPhysicalDeviceSubgroupProperties subgroup;
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control;
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_host;
        VkPhysicalDeviceDriverPropertiesKHR driver_properties;
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_properties;
    } properties;
    struct {
        bool is_moltenvk;
    } implementation;
} ShadyVkrPhysicalDeviceCaps;

bool shd_rt_check_physical_device_suitability(VkPhysicalDevice physical_device, ShadyVkrPhysicalDeviceCaps* out);

#endif
