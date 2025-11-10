#include <shady/ir/type.h>

#include "shady/runtime/vulkan.h"

#include "shady/ir/module.h"
#include "shady/ir/grammar.h"
#include "shady/ir/annotation.h"
#include "shady/ir/int.h"

#include "portability.h"
#include "log.h"
#include "util.h"

void shd_rt_vk_get_entry_point_interface(const Node* decl, size_t* count, RuntimeInterfaceItem* out) {
    *count = 0;

    for (size_t i = 0; i < decl->annotations.count; i++) {
        const Node* interface = decl->annotations.nodes[i];
        if (!shd_string_starts_with(get_annotation_name(interface), "EntryPointInterface"))
            continue;

        assert(interface->tag == Annotation_TAG);
        assert(interface->annotations.count == 2);
        const Node* src = interface->annotations.nodes[0];
        const Node* dst = interface->annotations.nodes[1];

        if (out) {
            if (strcmp(get_annotation_name(dst), "DstPushConstant") == 0) {
                out[*count].dst_kind = SHD_RII_Dst_PushConstant;
                out[*count].dst_details.push_constant.offset = shd_get_int_value(shd_get_annotation_values(dst).nodes[0], false);
                out[*count].dst_details.push_constant.size = shd_get_int_value(shd_get_annotation_values(dst).nodes[1], false);
            } else {
                shd_log_node(ERROR, dst);
                shd_error("Unknown interface destination");
            }

            if (strcmp(get_annotation_name(src), "SrcParam") == 0) {
                out[*count].src_kind = SHD_RII_Src_Param;
                out[*count].src_details.param.param_idx = shd_get_int_value(shd_get_annotation_value(src), false);
            } else if (strcmp(get_annotation_name(src), "SrcTmp") == 0) {
                out[*count].src_kind = SHD_RII_Src_TmpAllocation;
                out[*count].src_details.tmp_allocation.size = shd_get_annotation_value(src);
            } else if (strcmp(get_annotation_name(src), "SrcConstant") == 0) {
                out[*count].src_kind = SHD_RII_Src_LiftedConstant;
                out[*count].src_details.lifted_constant.constant = shd_get_annotation_value(src);
            } else if (strcmp(get_annotation_name(src), "SrcScratch") == 0) {
                out[*count].src_kind = SHD_RII_Src_ScratchBuffer;
                out[*count].src_details.scratch_buffer.per_invocation_size = shd_get_annotation_value(src);
            } else {
                shd_log_node(ERROR, src);
                shd_error("Unknown interface source");
            }
        }
        (*count)++;
    }
}


void shd_rt_vk_get_module_interface(Module* mod, size_t* count, RuntimeInterfaceItem* out) {
    Nodes decls = shd_module_get_all_exported(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != Function_TAG) continue;
        if (shd_lookup_annotation(decl, "EntryPoint"))
            return shd_rt_vk_get_entry_point_interface(decl, count, out);
    }
}