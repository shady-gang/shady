#include "vk_runner_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void bind_program_resources(VkrCommand* cmd, VkrSpecProgram* prog, int args_count, void** args) {
    LARRAY(VkWriteDescriptorSet, write_descriptor_sets, prog->interface_items_count);
    LARRAY(VkDescriptorBufferInfo, descriptor_buffer_info, prog->interface_items_count);
    size_t write_descriptor_sets_count = 0;

    LARRAY(VkDescriptorSet, bind_sets, prog->interface_items_count);
    size_t bind_sets_count = 0;

    size_t push_constant_size = shd_vkr_get_push_constant_size(prog);
    void* push_constant_buffer = calloc(push_constant_size, 1);

    for (size_t i = 0; i < prog->interface_items_count; i++) {
        RuntimeInterfaceItemEx* resource = &prog->interface_items[i];

        switch (resource->interface_item.dst_kind) {
            case SHD_RII_Dst_PushConstant: {
                switch (resource->interface_item.src_kind) {
                    case SHD_RII_Src_Param: {
                        size_t idx = resource->interface_item.src_details.param.param_idx;
                        assert(idx < args_count);
                        memcpy((uint8_t*) push_constant_buffer + resource->interface_item.dst_details.push_constant.offset, args[idx], resource->interface_item.dst_details.push_constant.size);
                        break;
                    }
                    case SHD_RII_Src_LiftedConstant: {
                        VkDeviceAddress bda = shd_rn_get_buffer_device_pointer((Buffer*) resource->buffer);
                        memcpy((uint8_t*) push_constant_buffer + resource->interface_item.dst_details.push_constant.offset,
                            &bda,
                            resource->interface_item.dst_details.push_constant.size);
                        assert(resource->interface_item.dst_details.push_constant.size == sizeof(VkDeviceAddress));
                        break;
                    }
                }
                break;
            }
            case SHD_RII_Dst_Descriptor: {
                // TODO
                /*descriptor_buffer_info[write_descriptor_sets_count] = (VkDescriptorBufferInfo) {
                    .buffer = resource->buffer->buffer,
                    .offset = resource->buffer->offset,
                    .range = resource->buffer->size - resource->buffer->offset,
                };*/

                write_descriptor_sets[write_descriptor_sets_count] = (VkWriteDescriptorSet) {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = NULL,
                    .descriptorType = resource->interface_item.dst_details.descriptor.type,
                    .descriptorCount = 1,
                    .dstSet = prog->sets[resource->interface_item.dst_details.descriptor.set],
                    .dstBinding = resource->interface_item.dst_details.descriptor.binding,
                    .pBufferInfo = &descriptor_buffer_info[write_descriptor_sets_count],
                };

                write_descriptor_sets_count++;
                break;
            }
        }
    }

    if (prog->required_descriptor_counts_count > 0) {
        vkUpdateDescriptorSets(prog->device->device, write_descriptor_sets_count, write_descriptor_sets, 0, NULL);
        for (size_t set = 0; set < MAX_DESCRIPTOR_SETS; set++) {
            bind_sets[set] = prog->sets[set];
        }
        bind_sets_count = MAX_DESCRIPTOR_SETS;
        vkCmdBindDescriptorSets(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, prog->layout, 0, bind_sets_count, bind_sets, 0, NULL);
    }

    vkCmdPushConstants(cmd->cmd_buf, prog->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constant_buffer);
    free(push_constant_buffer);
}

static Command make_command_base() {
    return (Command) {
        .wait_for_completion = (bool (*)(Command*)) shd_vkr_wait_completion,
    };
}

VkrCommand* shd_vkr_launch_kernel(VkrDevice* device, Program* program, String entry_point, int dimx, int dimy, int dimz, int args_count, void** args, ExtraKernelOptions* options) {
    assert(program && device);

    VkrSpecProgram* prog = shd_vkr_get_specialized_program(program, entry_point, device);

    shd_debug_print("Dispatching kernel on %s\n", device->caps.properties.base.properties.deviceName);

    VkrCommand* cmd = shd_vkr_begin_command(device);
    if (!cmd)
        return NULL;

    vkCmdBindPipeline(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, prog->pipeline);
    bind_program_resources(cmd, prog, args_count, args);

    if (options && options->profiled_gpu_time) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = NULL,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        CHECK_VK(vkCreateQueryPool(device->device, &qpci, NULL, &cmd->query_pool), {});
        cmd->profiled_gpu_time = options->profiled_gpu_time;
        vkCmdResetQueryPool(cmd->cmd_buf, cmd->query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd->cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, cmd->query_pool, 0);
    }

    vkCmdDispatch(cmd->cmd_buf, dimx, dimy, dimz);

    if (options && options->profiled_gpu_time) {
        vkCmdWriteTimestamp(cmd->cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cmd->query_pool, 1);
    }

    if (!shd_vkr_submit_command(cmd))
        goto err_post_commands_create;

    return cmd;

err_post_commands_create:
    shd_vkr_destroy_command(cmd);
    return NULL;
}

VkrCommand* shd_vkr_begin_command(VkrDevice* device) {
    VkrCommand* cmd = calloc(1, sizeof(VkrCommand));
    cmd->base = make_command_base();
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
    vkFreeCommandBuffers(device->device, device->cmd_pool, 1, &cmd->cmd_buf);
err_post_commands_create:
    free(cmd);
    return NULL;
}

bool shd_vkr_submit_command(VkrCommand* cmd) {
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

bool shd_vkr_wait_completion(VkrCommand* cmd) {
    assert(cmd->submitted && "Command must be submitted before they can be waited on");
    CHECK_VK(vkWaitForFences(cmd->device->device, 1, (VkFence[]) { cmd->done_fence }, true, UINT32_MAX), return false);
    if (cmd->profiled_gpu_time) {
        uint64_t ts[2];
        CHECK_VK(vkGetQueryPoolResults(cmd->device->device, cmd->query_pool, 0, 2, sizeof(uint64_t) * 2, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT), {});
        *cmd->profiled_gpu_time = (ts[1] - ts[0]) * cmd->device->caps.properties.base.properties.limits.timestampPeriod;
    }
    shd_vkr_destroy_command(cmd);
    return true;
}

void shd_vkr_destroy_command(VkrCommand* cmd) {
    if (cmd->submitted)
        vkDestroyFence(cmd->device->device, cmd->done_fence, NULL);
    if (cmd->query_pool)
        vkDestroyQueryPool(cmd->device->device, cmd->query_pool, NULL);
    vkFreeCommandBuffers(cmd->device->device, cmd->device->cmd_pool, 1, &cmd->cmd_buf);
    free(cmd);
}
