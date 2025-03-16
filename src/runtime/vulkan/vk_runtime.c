#include "shady/runtime/vulkan.h"

#include "shady/ir/module.h"
#include "shady/ir/grammar.h"
#include "shady/ir/annotation.h"
#include "shady/ir/int.h"

void shd_vkr_get_constant_data(Module* mod, size_t* count, size_t* sizes, void** data) {
    Nodes decls = shd_module_get_all_exported(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        if (decl->tag != GlobalVariable_TAG) continue;

        if (shd_lookup_annotation(decl, "Constants")) {
            AddressSpace as = decl->payload.global_variable.address_space;
            switch (as) {
                case AsShaderStorageBufferObject:
                case AsUniform: break;
                default: assert(false);
            }

            //int set = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(shd_lookup_annotation(decl, "DescriptorSet"))), false);
            //int binding = shd_get_int_literal_value(*shd_resolve_to_int_literal(shd_get_annotation_value(shd_lookup_annotation(decl, "DescriptorBinding"))), false);

            //ProgramResourceInfo* res_info = shd_arena_alloc(program->arena, sizeof(ProgramResourceInfo));
            //*res_info = (ProgramResourceInfo) {
            //    .is_bound = true,
            //    .as = as,
            //    .set = set,
            //    .binding = binding,
            //};
            //shd_growy_append_object(resources, res_info);
            //program->resources.num_resources++;

            const Type* struct_t = decl->payload.global_variable.type;
            assert(struct_t->tag == RecordType_TAG && struct_t->payload.record_type.special == DecorateBlock);

            for (size_t j = 0; j < struct_t->payload.record_type.members.count; j++) {
                const Type* member_t = struct_t->payload.record_type.members.nodes[j];
                assert(member_t->tag == PtrType_TAG);
                //member_t = shd_get_pointer_type_element(member_t);
                //TypeMemLayout layout = shd_get_mem_layout(shd_module_get_arena(program->specialized_module), member_t);
                //ProgramResourceInfo* constant_res_info = shd_arena_alloc(program->arena, sizeof(ProgramResourceInfo));
                //*constant_res_info = (ProgramResourceInfo) {
                //    .parent = res_info,
                //    .as = as,
                //};
                //shd_growy_append_object(resources, constant_res_info);
                //program->resources.num_resources++;
                //constant_res_info->size = layout.size_in_bytes;
                //constant_res_info->offset = res_info->size;
                //res_info->size += sizeof(void*);

                // TODO initial value
                Nodes annotations = decl->annotations;
                for (size_t k = 0; k < annotations.count; k++) {
                    const Node* a = annotations.nodes[k];
                    if ((strcmp(get_annotation_name(a), "InitialValue") == 0) && shd_resolve_to_int_literal(shd_first(shd_get_annotation_values(a)))->value == j) {
                        constant_res_info->default_data = calloc(1, layout.size_in_bytes);
                        write_value(constant_res_info->default_data, shd_get_annotation_values(a).nodes[1]);
                        //printf("wowie");
                    }
                }
            }

            if (shd_vkr_can_import_host_memory(program->device))
                res_info->host_backed_allocation = true;
            else
                res_info->staging = calloc(1, res_info->size);

            VkDescriptorSetLayoutBinding vk_binding = {
                .binding = binding,
                .descriptorType = shd_vkr_as_to_descriptor_type(as),
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = NULL,
            };
            register_required_descriptors(program, &vk_binding);
            add_binding(layout_create_infos, bindings_lists, set, vk_binding);
        }
    }
}