#include "runtime_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void bind_program_resources(Command* cmd, SpecProgram* prog) {
    LARRAY(VkWriteDescriptorSet, write_descriptor_sets, prog->resources.num_resources);
    size_t write_descriptor_sets_count = 0;

    LARRAY(VkDescriptorSet, bind_sets, prog->resources.num_resources);
    size_t bind_sets_count = 0;

    for (size_t i = 0; i < prog->resources.num_resources; i++) {
        ProgramResourceInfo* resource = prog->resources.resources[i];
        if (resource->is_bound) {
            write_descriptor_sets[write_descriptor_sets_count++] = (VkWriteDescriptorSet) {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = NULL,
                .descriptorType = as_to_descriptor_type(resource->as),
                .descriptorCount = 1,
                .dstSet = prog->sets[resource->set],
                .dstBinding = resource->binding,
                .pBufferInfo = &(VkDescriptorBufferInfo) {
                    .buffer = resource->buffer->buffer,
                    .offset = resource->buffer->offset,
                    .range = resource->buffer->size - resource->buffer->offset,
                }
            };
        }
    }

    vkUpdateDescriptorSets(prog->device->device, write_descriptor_sets_count, write_descriptor_sets, 0, NULL);

    for (size_t set = 0; set < MAX_DESCRIPTOR_SETS; set++) {
        bind_sets[set] = prog->sets[set];
    }
    bind_sets_count = MAX_DESCRIPTOR_SETS;

    vkCmdBindDescriptorSets(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, prog->layout, 0, bind_sets_count, bind_sets, 0, NULL);
}

Command* launch_kernel(Program* program, Device* device, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args) {
    assert(program && device);

    SpecProgram* prog = get_specialized_program(program, entry_point, device);

    debug_print("Dispatching kernel on %s\n", device->caps.properties.base.properties.deviceName);

    Command* cmd = begin_command(device);
    if (!cmd)
        return NULL;

    ProgramParamsInfo entrypoint_info = prog->parameters;
    if (entrypoint_info.args_size) {
        assert(args_count == entrypoint_info.num_args && "number of arguments must match number of entrypoint arguments");

        size_t push_constant_buffer_size = entrypoint_info.args_size;
        LARRAY(unsigned char, push_constant_buffer, push_constant_buffer_size);
        for (int i = 0; i < entrypoint_info.num_args; ++i)
            memcpy(push_constant_buffer + entrypoint_info.arg_offset[i], args[i], entrypoint_info.arg_size[i]);

        vkCmdPushConstants(cmd->cmd_buf, prog->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_buffer_size, push_constant_buffer);
    }

    vkCmdBindPipeline(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, prog->pipeline);
    bind_program_resources(cmd, prog);
    vkCmdDispatch(cmd->cmd_buf, dimx, dimy, dimz);

    if (!submit_command(cmd))
        goto err_post_commands_create;

    return cmd;

err_post_commands_create:
    destroy_command(cmd);
    return NULL;
}

Command* begin_command(Device* device) {
    Command* cmd = calloc(1, sizeof(Command));
    cmd->device = device;

    CHECK_VK(vkAllocateCommandBuffers(device->device, &(VkCommandBufferAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    }, &cmd->cmd_buf), goto err_post_commands_create);

    CHECK_VK(vkBeginCommandBuffer(cmd->cmd_buf, &(VkCommandBufferBeginInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL
    }), goto err_post_cmd_buf_create);

    return cmd;

err_post_cmd_buf_create:
    vkFreeCommandBuffers(device, device->cmd_pool, 1, &cmd->cmd_buf);
err_post_commands_create:
    free(cmd);
    return NULL;
}

bool submit_command(Command* cmd) {
    CHECK_VK(vkEndCommandBuffer(cmd->cmd_buf), return false);

    CHECK_VK(vkCreateFence(cmd->device->device, &(VkFenceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0
    }, NULL, &cmd->done_fence), return false);

    CHECK_VK(vkQueueSubmit(cmd->device->compute_queue, 1, &(VkSubmitInfo) {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = (VkCommandBuffer[]) { cmd->cmd_buf },
        .signalSemaphoreCount = 0
    }, cmd->done_fence), goto err_post_fence_create);

    cmd->submitted = true;

    return true;

err_post_fence_create:
    vkDestroyFence(cmd->device->device, cmd->done_fence, NULL);
    return false;
}

bool wait_completion(Command* cmd) {
    assert(cmd->submitted && "Command must be submitted before they can be waited on");
    CHECK_VK(vkWaitForFences(cmd->device->device, 1, (VkFence[]) { cmd->done_fence }, true, UINT32_MAX), return false);
    destroy_command(cmd);
    return true;
}

void destroy_command(Command* cmd) {
    if (cmd->submitted)
        vkDestroyFence(cmd->device->device, cmd->done_fence, NULL);
    vkFreeCommandBuffers(cmd->device->device, cmd->device->cmd_pool, 1, &cmd->cmd_buf);
    free(cmd);
}
