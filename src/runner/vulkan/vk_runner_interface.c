#include <shady/runtime/runtime.h>

#include "vk_runner_private.h"

void shd_vkr_populate_interface(VkrSpecProgram* spec) {
    assert(spec);
    shd_vkr_get_runtime_dependencies(spec->specialized_module, &spec->interface_items_count, NULL);
    LARRAY(RuntimeInterfaceItem, base_items, spec->interface_items_count);
    shd_vkr_get_runtime_dependencies(spec->specialized_module, &spec->interface_items_count, base_items);
    LARRAY(RuntimeInterfaceItemEx, items, spec->interface_items_count);
    spec->interface_items = calloc(sizeof(RuntimeInterfaceItemEx), spec->interface_items_count);
    for (size_t i = 0; i < spec->interface_items_count; i++) {
        spec->interface_items[i].interface_item = base_items[i];
    }
}

static size_t max(size_t a, size_t b) {
    if (a > b) return a;
    return b;
}

size_t shd_vkr_get_push_constant_size(VkrSpecProgram* program) {
    size_t push_constant_size = 0;
    for (size_t i = 0; i < program->interface_items_count; i++) {
        RuntimeInterfaceItemEx item = program->interface_items[i];
        switch (item.interface_item.dst_kind) {
            case SHD_RII_Dst_PushConstant:
                push_constant_size = max(push_constant_size, item.interface_item.dst_details.push_constant.offset + item.interface_item.dst_details.push_constant.size);
            break;
            default: continue;
        }
    }
    return push_constant_size;
}