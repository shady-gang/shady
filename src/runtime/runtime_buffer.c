#include "runtime_private.h"

#include "log.h"

typedef enum {
    AllocDeviceLocal,
    AllocHostVisible
} AllocHeap;

static uint32_t find_suitable_memory_type(Device* device, uint32_t memory_type_bits, AllocHeap heap) {
    VkPhysicalDeviceMemoryProperties device_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(device->caps.physical_device, &device_memory_properties);
    for (size_t bit = 0; bit < 32; bit++) {
        VkMemoryType memory_type = device_memory_properties.memoryTypes[bit];

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

static Buffer* create_buffer_internal(Device* device, void* imported_ptr, size_t size) {
    Buffer* buffer = calloc(sizeof(Buffer), 1);
    buffer->imported = imported_ptr != NULL;

    VkBufferCreateInfo buffer_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .size = size,
        .flags = 0,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };
    if (device->caps.features.buffer_device_address.bufferDeviceAddress)
        buffer_create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT;
    CHECK_VK(vkCreateBuffer(device->device, &buffer_create_info, NULL, &buffer->buffer), goto bail_out);
    VkMemoryAllocateInfo allocation_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = (VkDeviceSize) size, // the driver might want padding !
        .memoryTypeIndex = 0 /* set later */,
    };

    VkImportMemoryHostPointerInfoEXT import_host_ptr_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = NULL,
        .pHostPointer = imported_ptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
    };

    if (imported_ptr) {
        VkMemoryHostPointerPropertiesEXT host_ptr_properties = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
            .pNext = NULL
        };
        CHECK_VK(device->extensions.external_memory_host.vkGetMemoryHostPointerPropertiesEXT(device->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, imported_ptr, &host_ptr_properties), return NULL);
        allocation_info.memoryTypeIndex = find_suitable_memory_type(device, host_ptr_properties.memoryTypeBits, AllocHostVisible);
        append_pnext((VkBaseOutStructure*) &allocation_info, &import_host_ptr_info);
    } else {
        VkBufferMemoryRequirementsInfo2 buf_mem_requirements = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
            .pNext = NULL,
            .buffer = buffer->buffer
        };
        VkMemoryRequirements2 mem_requirements = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
            .pNext = NULL,
        };
        vkGetBufferMemoryRequirements2(device->device, &buf_mem_requirements, &mem_requirements);
        allocation_info.memoryTypeIndex = find_suitable_memory_type(device, mem_requirements.memoryRequirements.memoryTypeBits, AllocDeviceLocal);
    }

    VkMemoryAllocateFlagsInfo allocate_flags =  {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = NULL,
        .flags = 0,
        .deviceMask = 0
    };
    if (device->caps.features.buffer_device_address.bufferDeviceAddress)
        allocate_flags.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    append_pnext((VkBaseOutStructure*) &allocation_info, &allocate_flags);

    CHECK_VK(vkAllocateMemory(device->device, &allocation_info, NULL, &buffer->memory), goto bail_out);
    vkBindBufferMemory(device->device, buffer->buffer, buffer->memory, 0);
    return buffer;

    bail_out:
    free(buffer);
    return NULL;
}

Buffer* allocate_buffer_device(Device* device, size_t size) {
    return create_buffer_internal(device, NULL, size);
}

Buffer* import_buffer_host(Device* device, void* ptr, size_t size) {
    return create_buffer_internal(device, ptr, size);
}

void destroy_buffer(Buffer* buffer) {
    vkDestroyBuffer(buffer->device->device, buffer->buffer, NULL);
    if (!buffer->imported)
        vkFreeMemory(buffer->device->device, buffer->memory, NULL);
}

bool copy_into_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size);
bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size);
