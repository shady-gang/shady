#include "runtime_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

Commands* launch_kernel(Program* program, Device* device, int dimx, int dimy, int dimz, int args_count, void** args) {
    assert(program && device);

    SpecProgram* prog = get_specialized_program(program, device);

    debug_print("Dispatching kernel on %s\n", device->caps.properties.base.properties.deviceName);

    Commands* commands = begin_commands(device);
    if (!commands)
        return NULL;

    EntryPointInfo entrypoint_info = prog->entrypoint;
    if (entrypoint_info.args_size) {
        assert(args_count == entrypoint_info.num_args && "number of arguments must match number of entrypoint arguments");

        size_t push_constant_buffer_size = entrypoint_info.args_size;
        LARRAY(unsigned char, push_constant_buffer, push_constant_buffer_size);
        for (int i = 0; i < entrypoint_info.num_args; ++i)
            memcpy(push_constant_buffer + entrypoint_info.arg_offset[i], args[i], entrypoint_info.arg_size[i]);

        vkCmdPushConstants(commands->cmd_buf, prog->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_buffer_size, push_constant_buffer);
    }

    vkCmdBindPipeline(commands->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, prog->pipeline);
    vkCmdDispatch(commands->cmd_buf, dimx, dimy, dimz);

    if (!submit_commands(commands))
        goto err_post_commands_create;

    return commands;

err_post_commands_create:
    destroy_commands(commands);
    return NULL;
}

Commands* begin_commands(Device* device) {
    Commands* commands = calloc(1, sizeof(Commands));
    commands->device = device;

    CHECK_VK(vkAllocateCommandBuffers(device->device, &(VkCommandBufferAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    }, &commands->cmd_buf), goto err_post_commands_create);

    CHECK_VK(vkBeginCommandBuffer(commands->cmd_buf, &(VkCommandBufferBeginInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL
    }), goto err_post_cmd_buf_create);

    return commands;

err_post_cmd_buf_create:
    vkFreeCommandBuffers(device, device->cmd_pool, 1, &commands->cmd_buf);
err_post_commands_create:
    free(commands);
    return NULL;
}

bool submit_commands(Commands* commands) {
    CHECK_VK(vkEndCommandBuffer(commands->cmd_buf), return false);

    CHECK_VK(vkCreateFence(commands->device->device, &(VkFenceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0
    }, NULL, &commands->done_fence), return false);

    CHECK_VK(vkQueueSubmit(commands->device->compute_queue, 1, &(VkSubmitInfo) {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = (VkCommandBuffer[]) { commands->cmd_buf },
        .signalSemaphoreCount = 0
    }, commands->done_fence), goto err_post_fence_create);

    commands->submitted = true;

    return true;

err_post_fence_create:
    vkDestroyFence(commands->device->device, commands->done_fence, NULL);
    return false;
}

bool wait_completion(Commands* commands) {
    assert(commands->submitted && "Commands must be submitted before they can be waited on");
    CHECK_VK(vkWaitForFences(commands->device->device, 1, (VkFence[]) { commands->done_fence }, true, UINT32_MAX), return false);
    destroy_commands(commands);
    return true;
}

void destroy_commands(Commands* commands) {
    if (commands->submitted)
        vkDestroyFence(commands->device->device, commands->done_fence, NULL);
    vkFreeCommandBuffers(commands->device->device, commands->device->cmd_pool, 1, &commands->cmd_buf);
    free(commands);
}
