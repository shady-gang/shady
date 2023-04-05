#include "runtime_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef enum { DispatchCompute } DispatchType;

struct Dispatch_ {
    DispatchType type;
    SpecProgram* src;

    VkCommandBuffer cmd_buf;
    VkFence done_fence;
};

Dispatch* launch_kernel(Program* program, Device* device, int dimx, int dimy, int dimz, int args_count, void** args) {
    assert(program && device);

    Dispatch* dispatch = calloc(1, sizeof(Dispatch));
    dispatch->type = DispatchCompute;
    dispatch->src = get_specialized_program(program, device);

    debug_print("Dispatching kernel on %s\n", device->caps.properties.base.properties.deviceName);

    CHECK_VK(vkAllocateCommandBuffers(device->device, &(VkCommandBufferAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    }, &dispatch->cmd_buf), return NULL);

    CHECK_VK(vkBeginCommandBuffer(dispatch->cmd_buf, &(VkCommandBufferBeginInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL
    }), return NULL);

    EntryPointInfo entrypoint_info = dispatch->src->entrypoint;
    if (entrypoint_info.args_size) {
        assert(args_count == entrypoint_info.num_args && "number of arguments must match number of entrypoint arguments");

        size_t push_constant_buffer_size = entrypoint_info.args_size;
        LARRAY(unsigned char, push_constant_buffer, push_constant_buffer_size);
        for (int i = 0; i < entrypoint_info.num_args; ++i)
            memcpy(push_constant_buffer + entrypoint_info.arg_offset[i], args[i], entrypoint_info.arg_size[i]);

        vkCmdPushConstants(dispatch->cmd_buf, dispatch->src->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_buffer_size, push_constant_buffer);
    }

    vkCmdBindPipeline(dispatch->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch->src->pipeline);
    vkCmdDispatch(dispatch->cmd_buf, dimx, dimy, dimz);

    CHECK_VK(vkEndCommandBuffer(dispatch->cmd_buf), return NULL);

    CHECK_VK(vkCreateFence(device->device, &(VkFenceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0
    }, NULL, &dispatch->done_fence), return NULL);

    CHECK_VK(vkQueueSubmit(device->compute_queue, 1, &(VkSubmitInfo) {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = (VkCommandBuffer[]) { dispatch->cmd_buf },
        .signalSemaphoreCount = 0
    }, dispatch->done_fence), return NULL);

    return dispatch;
}

bool wait_completion(Dispatch* dispatch) {
    VkDevice device = dispatch->src->device->device;
    CHECK_VK(vkWaitForFences(device, 1, (VkFence[]) { dispatch->done_fence }, true, UINT32_MAX), return false);

    vkDestroyFence(device, dispatch->done_fence, NULL);
    vkFreeCommandBuffers(device, dispatch->src->device->cmd_pool, 1, &dispatch->cmd_buf);

    free(dispatch);
    return true;
}
