#ifndef SHADY_RUNNER_VK_H
#define SHADY_RUNNER_VK_H

#include "vulkan/vulkan.h"
#include "shady/runtime/vulkan.h"

void shd_rn_provide_vkinstance(VkInstance);

Device* shd_rn_open_vkdevice_with_caps(Runner*, ShadyVkrPhysicalDeviceCaps, VkDevice);
Device* shd_rn_open_vkdevice(Runner*, VkPhysicalDevice, VkDevice);

VkDevice shd_rn_get_vkdevice(Device*);
VkBuffer shd_rn_get_vkbuffer(Buffer*);

#endif