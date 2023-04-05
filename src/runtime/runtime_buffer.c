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
    size_t offset;
    void* host_ptr;
};

static Buffer* create_buffer_internal(Device* device, void* imported_ptr, size_t size) {
    Buffer* buffer = calloc(sizeof(Buffer), 1);
    buffer->device = device;
    buffer->imported = imported_ptr != NULL;
    buffer->offset = 0;

    VkBufferCreateInfo buffer_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .size = size,
        .flags = 0,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT /** we basically want all the usages ! */
    };

    VkExternalMemoryBufferCreateInfo ext_memory_buffer_create_info;
    if (imported_ptr) {
        // align the bugger first ...
        size_t desired_alignment = device->caps.properties.external_memory_host.minImportedHostPointerAlignment;
        size_t unaligned_addr = (size_t) imported_ptr;
        size_t aligned_addr = (unaligned_addr / desired_alignment) * desired_alignment;
        assert(unaligned_addr >= aligned_addr);
        buffer->offset = unaligned_addr - aligned_addr;
        warn_print("desired alignment = %zu, offset = %zu\n", desired_alignment, buffer->offset);

        size_t unaligned_end = unaligned_addr + size;
        assert(unaligned_end >= aligned_addr);
        size_t aligned_end = ((unaligned_end + desired_alignment - 1) / desired_alignment) * desired_alignment;
        assert(aligned_end >= unaligned_end);
        size_t aligned_size = aligned_end - aligned_addr;
        assert(aligned_size >= size);
        assert(aligned_size % desired_alignment == 0);
        warn_print("unaligned start %zu end %zu\n", unaligned_addr, unaligned_end);
        warn_print("aligned start %zu end %zu\n", aligned_addr, aligned_end);

        imported_ptr = (void*) aligned_addr;
        size = aligned_size;

        ext_memory_buffer_create_info = (VkExternalMemoryBufferCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                .pNext = NULL,
        };
        append_pnext((VkBaseOutStructure*) &buffer_create_info, &ext_memory_buffer_create_info);
    }

    // create buffer to use that memory
    if (device->caps.features.buffer_device_address.bufferDeviceAddress)
        buffer_create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT;
    CHECK_VK(vkCreateBuffer(device->device, &buffer_create_info, NULL, &buffer->buffer), goto err_post_obj_create);

    VkMemoryAllocateInfo allocation_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize =  0 /* set later */,
        .memoryTypeIndex = 0 /* set later */,
    };

    VkImportMemoryHostPointerInfoEXT import_host_ptr_info;
    size_t memory_bind_offset = 0;

    if (imported_ptr) {
        VkMemoryHostPointerPropertiesEXT host_ptr_properties = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
            .pNext = NULL
        };
        CHECK_VK(device->extensions.EXT_external_memory_host.vkGetMemoryHostPointerPropertiesEXT(device->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void*) imported_ptr, &host_ptr_properties), goto err_post_buffer_create);
        uint32_t memory_type_index = find_suitable_memory_type(device, host_ptr_properties.memoryTypeBits, AllocHostVisible);
        VkPhysicalDeviceMemoryProperties device_memory_properties;
        vkGetPhysicalDeviceMemoryProperties(device->caps.physical_device, &device_memory_properties);
        warn_print("memory type index: %d heap: %d\n", memory_type_index, device_memory_properties.memoryTypes[memory_type_index].heapIndex);
        allocation_info.allocationSize = (VkDeviceSize) size;
        allocation_info.memoryTypeIndex = memory_type_index;

        import_host_ptr_info = (VkImportMemoryHostPointerInfoEXT) {
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            .pNext = NULL,
            .pHostPointer = (void*) imported_ptr,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
        };
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
        allocation_info.allocationSize = mem_requirements.memoryRequirements.size;
        allocation_info.memoryTypeIndex = find_suitable_memory_type(device, mem_requirements.memoryRequirements.memoryTypeBits, AllocDeviceLocal);
    }

    // Add extra allocation flags
    VkMemoryAllocateFlagsInfo allocate_flags =  {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = NULL,
        .flags = 0,
        .deviceMask = 0
    };
    if (device->caps.features.buffer_device_address.bufferDeviceAddress)
        allocate_flags.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    append_pnext((VkBaseOutStructure*) &allocation_info, &allocate_flags);

    // allocate memory
    CHECK_VK(vkAllocateMemory(device->device, &allocation_info, NULL, &buffer->memory), goto err_post_buffer_create);

    // bind the two together
    CHECK_VK(vkBindBufferMemory(device->device, buffer->buffer, buffer->memory, memory_bind_offset), goto err_post_mem_alloc);

    //if (imported_ptr) {
    //    CHECK_VK(vkMapMemory(device->device, buffer->memory, 0, size, 0, &buffer->host_ptr), goto err_post_mem_alloc);
    //}

    return buffer;

err_post_mem_alloc:
    vkFreeMemory(buffer->device->device, buffer->memory, NULL);
err_post_buffer_create:
    vkDestroyBuffer(buffer->device->device, buffer->buffer, NULL);
err_post_obj_create:
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
    vkFreeMemory(buffer->device->device, buffer->memory, NULL);
}

VkDeviceAddress get_buffer_device_pointer(Buffer* buf) {
    return vkGetBufferDeviceAddress(buf->device->device, &(VkBufferDeviceAddressInfo) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = NULL,
        .buffer = buf->buffer
    }) + buf->offset;
}

void* get_buffer_host_pointer(Buffer* buf) {
    return ((char*) buf->host_ptr) + buf->offset;
}

bool resize_staging_buffer(Device* device, size_t size) {
    if (size <= device->staging_buffer.size)
        return true;

    VkBufferCreateInfo buffer_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .size = size,
        .flags = 0,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    };
    CHECK_VK(vkCreateBuffer(device->device, &buffer_create_info, NULL, &device->staging_buffer.buffer), return false);

    VkBufferMemoryRequirementsInfo2 buf_mem_requirements = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = NULL,
        .buffer = device->staging_buffer.buffer
    };
    VkMemoryRequirements2 mem_requirements = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetBufferMemoryRequirements2(device->device, &buf_mem_requirements, &mem_requirements);

    VkMemoryAllocateInfo allocation_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = NULL,
        .allocationSize = mem_requirements.memoryRequirements.size,
        .memoryTypeIndex = find_suitable_memory_type(device, mem_requirements.memoryRequirements.memoryTypeBits, AllocHostVisible)
    };
    CHECK_VK(vkAllocateMemory(device->device, &allocation_info, NULL, &device->staging_buffer.memory), goto err_post_buffer_create);

    CHECK_VK(vkBindBufferMemory(device->device, device->staging_buffer.buffer, device->staging_buffer.memory, 0), goto err_post_mem_alloc);

    CHECK_VK(vkMapMemory(device->device, device->staging_buffer.memory, 0, size, 0, &device->staging_buffer.ptr), goto err_post_mem_alloc);

    device->staging_buffer.size = size;

    return true;

err_post_mem_alloc:
    vkFreeMemory(device, device->staging_buffer.memory, NULL);
err_post_buffer_create:
    vkDestroyBuffer(device, device->staging_buffer.buffer, NULL);
    return false;
}

void free_staging_buffer(Device* device) {
    if (device->staging_buffer.size) {
        vkDestroyBuffer(device->device, device->staging_buffer.buffer, NULL);
        vkFreeMemory(device->device, device->staging_buffer.memory, NULL);
    }
}

static Command* submit_buffer_copy(Device* device, VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size) {
    Command* commands = begin_command(device);
    if (!commands)
        return NULL;

    vkCmdCopyBuffer(commands->cmd_buf, src, dst, 1, (VkBufferCopy[]) { { .srcOffset = src_offset, .dstOffset = dst_offset, .size = size } });

    if (!submit_command(commands))
        goto err_post_commands_create;

    return commands;

err_post_commands_create:
    destroy_command(commands);
    return NULL;
}

bool copy_to_buffer(Buffer* dst, size_t buffer_offset, void* src, size_t size) {
    Device* device = dst->device;

    Buffer* src_buf = import_buffer_host(device, src, size);
    if (!src_buf)
        return false;

    if (!wait_completion(submit_buffer_copy(device, src_buf->buffer, src_buf->offset, dst->buffer, dst->offset + buffer_offset, size)))
        goto err_post_buffer_import;

    destroy_buffer(src_buf);
    return true;

err_post_buffer_import:
    destroy_buffer(src_buf);
    return false;
}

bool copy_from_buffer(Buffer* src, size_t buffer_offset, void* dst, size_t size) {
    Device* device = src->device;

    Buffer* dst_buf = import_buffer_host(device, dst, size);
    if (!dst_buf)
        return false;

    if (!wait_completion(submit_buffer_copy(device, src->buffer, src->offset + buffer_offset, dst_buf->buffer, dst_buf->offset, size)))
        goto err_post_buffer_import;

    destroy_buffer(dst_buf);
    return true;

err_post_buffer_import:
    destroy_buffer(dst_buf);
    return false;
}
