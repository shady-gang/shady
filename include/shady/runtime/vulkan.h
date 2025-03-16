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
    } src_kind;
    union {
        struct {
            size_t param_idx;
        } param;
        struct {
            const Node* constant;
        } lifted_constant;
    } src_details;
} RuntimeInterfaceItem;

void shd_vkr_get_runtime_dependencies(Module*, size_t* count, RuntimeInterfaceItem* out);

typedef struct {
    VkDescriptorSetLayoutCreateInfo set_layout;
} ShdDescriptorSetLayout;

void shd_vkr_get_descriptor_layouts(Module*, size_t* count, ShdDescriptorSetLayout** out);
void shd_vkr_free_descriptor_set_layouts();

void shd_vkr_write_push_constants(Module*, size_t arguments_count, void** arguments);

#endif
