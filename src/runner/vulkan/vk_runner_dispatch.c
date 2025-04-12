#include "vk_runner_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void prepare_resources_for_launch(VkrCommand* cmd, VkrSpecProgram* prog, int threadsx, int threadsy, int threadsz, int args_count, void** args) {
    LARRAY(VkWriteDescriptorSet, write_descriptor_sets, prog->interface_items_count);
    LARRAY(VkDescriptorBufferInfo, descriptor_buffer_info, prog->interface_items_count);
    size_t write_descriptor_sets_count = 0;

    LARRAY(VkDescriptorSet, bind_sets, prog->interface_items_count);
    size_t bind_sets_count = 0;

    size_t push_constant_size = shd_vkr_get_push_constant_size(prog);
    void* push_constant_buffer = calloc(push_constant_size, 1);

    for (size_t i = 0; i < prog->interface_items_count; i++) {
        VkrProgramInterfaceItem* resource = &prog->interface_items[i];
        VkrDispatchInterfaceItem* dispatch_item = &cmd->launch_interface_items[i];

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
                    case SHD_RII_Src_ScratchBuffer: {
                        size_t threads = threadsx * threadsy * threadsz;
                        size_t total_size = threads * resource->per_invocation_size;

                        //printf("blocks: %zu wg: %zu, total: %zu\n", blocks, threads_per_wg, total_size);
                        if (!resource->scratch || resource->scratch_size != total_size) {
                            if (resource->scratch_size != 0)
                                shd_vkr_destroy_buffer(resource->scratch);
                            resource->scratch = shd_vkr_allocate_buffer_device(cmd->device, total_size);
                            resource->scratch_size = total_size;
                        }
                        //char* zeroes = calloc(total_size, 1);
                        //shd_rn_copy_to_buffer((Buffer*) dispatch_item->scratch, 0, zeroes, total_size);

                        VkDeviceAddress bda = shd_rn_get_buffer_device_pointer((Buffer*) resource->scratch);
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
        vkCmdBindDescriptorSets(cmd->cmd_buf, prog->bind_point, prog->layout, 0, bind_sets_count, bind_sets, 0, NULL);
    }

    vkCmdPushConstants(cmd->cmd_buf, prog->layout, prog->stage, 0, push_constant_size, push_constant_buffer);
    free(push_constant_buffer);
}

static void cleanup_resources_after_launch(VkrCommand* command) {
    for (size_t i = 0; i < command->launched_program->interface_items_count; i++) {
        VkrDispatchInterfaceItem* resource = &command->launch_interface_items[i];
    }
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

    cmd->launched_program = prog;
    cmd->launch_interface_items = calloc(sizeof(VkrDispatchInterfaceItem), prog->interface_items_count);

    vkCmdBindPipeline(cmd->cmd_buf, prog->bind_point, prog->pipeline);
    //size_t blocks = dimx * dimy * dimz;

    //const Node* ep = shd_module_get_exported(prog->specialized_module, prog->key.entry_point);
    //assert(ep);
    //const Node* wgs = shd_lookup_annotation(ep, "WorkgroupSize");
    //assert(wgs);
    //Nodes values = shd_get_annotation_values(wgs);
    //assert(values.count == 3);
    //uint32_t wg_x_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[0]), false);
    //uint32_t wg_y_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[1]), false);
    //uint32_t wg_z_dim = (uint32_t) shd_get_int_literal_value(*shd_resolve_to_int_literal(values.nodes[2]), false);
    int threadsx = dimx * shd_get_arena_config(shd_module_get_arena(prog->specialized_module))->specializations.workgroup_size[0];
    int threadsy = dimy * shd_get_arena_config(shd_module_get_arena(prog->specialized_module))->specializations.workgroup_size[1];
    int threadsz = dimz * shd_get_arena_config(shd_module_get_arena(prog->specialized_module))->specializations.workgroup_size[2];
    prepare_resources_for_launch(cmd, prog, threadsx, threadsy, threadsz, args_count, args);

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

static VkrCommand* vkr_launch_rays(VkrDevice* device, Program* program, String entry_point, int sizex, int sizey, int sizez, int args_count, void** args, ExtraKernelOptions* options) {
    assert(program && device);

    VkrSpecProgram* prog = shd_vkr_get_specialized_program(program, entry_point, device);

    shd_debug_print("Dispatching rays on %s\n", device->caps.properties.base.properties.deviceName);

    VkrCommand* cmd = shd_vkr_begin_command(device);
    if (!cmd)
        return NULL;

    cmd->launched_program = prog;
    cmd->launch_interface_items = calloc(sizeof(VkrDispatchInterfaceItem), prog->interface_items_count);

    vkCmdBindPipeline(cmd->cmd_buf, prog->bind_point, prog->pipeline);
    prepare_resources_for_launch(cmd, prog, sizex, sizey, sizez, args_count, args);

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

    VkStridedDeviceAddressRegionKHR empty_sbt = {
        .deviceAddress = 0,
    };

    device->extensions.vkCmdTraceRaysKHR(cmd->cmd_buf, &prog->rt.rg_sbt, &empty_sbt, &empty_sbt, &prog->rt.callables_sbt, sizex, sizey, sizez);
    //device->extensions.vkCmdTraceRaysKHR(cmd->cmd_buf, &empty_sbt, &empty_sbt, &empty_sbt, &empty_sbt, 1, 1, 1);

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

Command* shd_vkr_launch_rays(Program* p, Device* d, const char* entry_point, int x, int y, int z, int args_count, void** args, ExtraKernelOptions* extra_options) {
    assert(d->backend == VulkanRuntimeBackend);
    return (Command*) vkr_launch_rays((VkrDevice*) d, (Program*) p, entry_point, x, y, z, args_count, args, extra_options);
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
    if (cmd->launch_interface_items)
        cleanup_resources_after_launch(cmd);
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
