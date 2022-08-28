#include "runtime_private.h"

#include "log.h"

typedef enum {
    AllocDeviceLocal,
    AllocHostVisible
} AllocHeap;

static uint32_t find_suitable_memory_type(Device* device, uint32_t memory_type_bits, AllocHeap heap) {
    VkPhysicalDeviceMemoryProperties device_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(device->properties.physical_device, &device_memory_properties);
    for (size_t bit = 0; bit < 32; bit++) {
        VkMemoryType memory_type = device_memory_properties.memoryTypes[bit];
        VkMemoryHeap memory_heap = device_memory_properties.memoryHeaps[memory_type.heapIndex];

        bool is_host_visible = (memory_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        bool is_host_coherent = (memory_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        bool is_device_local = (memory_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

        if ((memory_type_bits & (1 << bit)) != 0) {
            switch (heap) {
                case AllocDeviceLocal:
                    if (is_device_local)
                        return bit;
                    break;
                case AllocHostVisible:
                    if (is_host_visible && is_host_coherent)
                        return bit;
                    break;
            }
        }
    }
    assert(false && "Unable to find a suitable memory type");
}

struct Buffer_ {
    Device* device;
    bool imported;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

Buffer* allocate_buffer_device(Device* device, size_t size) {
    Buffer* buffer = calloc(sizeof(Buffer), 1);
    buffer->imported = false;

    VkBufferCreateInfo buffer_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .size = size,
        .flags = 0,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };
    if (device->properties.features.physical_global_ptrs)
        buffer_create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    CHECK_VK(vkCreateBuffer(device->device, &buffer_create_info, NULL, &buffer->buffer), goto bail_out);

    VkBufferMemoryRequirementsInfo2 buf_mem_requirements = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = NULL,
        .buffer = buffer->buffer
    };
    VkMemoryRequirements2 mem_requirements;
    vkGetBufferMemoryRequirements2(device->device, &buf_mem_requirements, &mem_requirements);

    VkMemoryAllocateFlagsInfo allocate_flags =  {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = NULL,
        .flags = 0,
        .deviceMask = 0
    };
    if (device->properties.features.physical_global_ptrs)
        allocate_flags.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo allocation_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = (VkDeviceSize) size, // the driver might want padding !
        .memoryTypeIndex = find_suitable_memory_type(device, mem_requirements.memoryRequirements.memoryTypeBits, AllocDeviceLocal),
    };
    append_pnext(&allocation_info, &allocate_flags);

    vkAllocateMemory(device->device, &allocation_info, NULL, &buffer->memory);
    vkBindBufferMemory(device->device, buffer->buffer, buffer->memory, 0);
    return buffer;

    bail_out:
    free(buffer);
    return NULL;
}

Buffer* import_buffer_host(Device* device, void* ptr, size_t size) {
    error("TODO");
}

Buffer* destroy_buffer(Buffer* buffer) {
    vkDestroyBuffer(buffer->device->device, buffer->buffer, NULL);
    if (!buffer->imported)
        vkFreeMemory(buffer->device->device, buffer->memory, NULL);
}

bool copy_into_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size);
bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size);
