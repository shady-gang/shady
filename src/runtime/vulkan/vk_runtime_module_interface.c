#include <shady/ir/type.h>

#include "shady/runtime/vulkan.h"

#include "shady/ir/module.h"
#include "shady/ir/grammar.h"
#include "shady/ir/annotation.h"
#include "shady/ir/int.h"
#include "shady/ir/memory_layout.h"

#include "portability.h"
#include "log.h"

void shd_vkr_get_runtime_dependencies(Module* mod, size_t* count, RuntimeInterfaceItem* out) {
    Nodes decls = shd_module_get_all_exported(mod);
    bool found = false;
    *count = 0;
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;

        if (shd_lookup_annotation(decl, "EntryPointPushConstants")) {
            assert(!found && "Two EntryPointPushConstants found");
            found = true;

            const Type* t = decl->payload.global_variable.type;
            assert(t->tag == StructType_TAG);
            StructType payload = t->payload.struct_type;
            LARRAY(FieldLayout, field_layouts, payload.members.count);
            shd_get_record_layout(t->arena, t, field_layouts);

            for (size_t j = 0; j < payload.members.count; j++) {
                Nodes annotations = decl->annotations;
                for (size_t k = 0; k < annotations.count; k++) {
                    const Node* an = annotations.nodes[k];
                    if (strcmp(get_annotation_name(an), "RuntimeProvideTmpAllocationInPushConstant") == 0) {
                        Nodes arr = an->payload.annotation_values.values;
                        size_t member_idx = shd_get_int_literal_value(*shd_resolve_to_int_literal(arr.nodes[0]), false);
                        if (member_idx != j)
                            continue;
                        const Node* size = arr.nodes[1];
                        if (out) {
                            TypeMemLayout layout = shd_get_mem_layout(payload.members.nodes[j]->arena, size_t_type(t->arena));
                            out[*count] = (RuntimeInterfaceItem) {
                                .dst_kind = SHD_RII_Dst_PushConstant,
                                .dst_details.push_constant = {
                                    .offset = field_layouts[j].offset_in_bytes,
                                    .size = layout.size_in_bytes,
                                },
                                .src_kind = SHD_RII_Src_TmpAllocation,
                                .src_details.tmp_allocation = {
                                    .size = size
                                },
                            };
                        }
                        (*count)++;
                        goto next;
                    } else if (strcmp(get_annotation_name(an), "RuntimeProvideConstantInPushConstant") == 0) {
                        Nodes arr = an->payload.annotation_values.values;
                        size_t member_idx = shd_get_int_literal_value(*shd_resolve_to_int_literal(arr.nodes[0]), false);
                        if (member_idx != j)
                            continue;
                        const Node* contents = arr.nodes[1];
                        if (out) {
                            TypeMemLayout layout = shd_get_mem_layout(payload.members.nodes[j]->arena, size_t_type(t->arena));
                            out[*count] = (RuntimeInterfaceItem) {
                                .dst_kind = SHD_RII_Dst_PushConstant,
                                .dst_details.push_constant = {
                                    .offset = field_layouts[j].offset_in_bytes,
                                    .size = layout.size_in_bytes,
                                },
                                .src_kind = SHD_RII_Src_LiftedConstant,
                                .src_details.lifted_constant = {
                                    .constant = contents
                                },
                            };
                        }
                        (*count)++;
                        goto next;
                    } else if (strcmp(get_annotation_name(an), "RuntimeProvideScratchInPushConstant") == 0) {
                        Nodes arr = an->payload.annotation_values.values;
                        size_t member_idx = shd_get_int_literal_value(*shd_resolve_to_int_literal(arr.nodes[0]), false);
                        if (member_idx != j)
                            continue;
                        const Node* contents = arr.nodes[1];
                        if (out) {
                            TypeMemLayout layout = shd_get_mem_layout(payload.members.nodes[j]->arena, size_t_type(t->arena));
                            out[*count] = (RuntimeInterfaceItem) {
                                .dst_kind = SHD_RII_Dst_PushConstant,
                                .dst_details.push_constant = {
                                    .offset = field_layouts[j].offset_in_bytes,
                                    .size = layout.size_in_bytes,
                                },
                                .src_kind = SHD_RII_Src_ScratchBuffer,
                                .src_details.scratch_buffer = {
                                    .per_invocation_size = contents
                                },
                            };
                        }
                        (*count)++;
                        goto next;
                    } else if (strcmp(get_annotation_name(an), "RuntimeParamInPushConstant") == 0) {
                        Nodes arr = an->payload.annotation_values.values;
                        size_t member_idx = shd_get_int_literal_value(*shd_resolve_to_int_literal(arr.nodes[0]), false);
                        if (member_idx != j)
                            continue;
                        size_t param_idx = shd_get_int_literal_value(*shd_resolve_to_int_literal(arr.nodes[1]), false);
                        if (out) {
                            TypeMemLayout layout = shd_get_mem_layout(payload.members.nodes[j]->arena, payload.members.nodes[j]);
                            out[*count] = (RuntimeInterfaceItem) {
                                .dst_kind = SHD_RII_Dst_PushConstant,
                                .dst_details.push_constant = {
                                    .offset = field_layouts[j].offset_in_bytes,
                                    .size = layout.size_in_bytes,
                                },
                                .src_kind = SHD_RII_Src_Param,
                                .src_details.param = {
                                    .param_idx = param_idx,
                                }
                            };
                        }
                        (*count)++;
                        goto next;
                    }
                }

                shd_error("Failed to find metadata for push constant field %lu\n", j);
                shd_error_die();
                next: continue;
            }
        }
    }
}
