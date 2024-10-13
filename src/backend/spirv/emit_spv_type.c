#include "emit_spv.h"

#include "shady/ir/memory_layout.h"

#include "shady/rewrite.h"

#include "portability.h"
#include "log.h"
#include "dict.h"

#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

SpvStorageClass spv_emit_addr_space(Emitter* emitter, AddressSpace address_space) {
    switch(address_space) {
        case AsShared:                       return SpvStorageClassWorkgroup;
        case AsPrivate:                      return SpvStorageClassPrivate;
        case AsFunction:                     return SpvStorageClassFunction;
        case AsGlobal:
            spvb_set_addressing_model(emitter->file_builder, SpvAddressingModelPhysicalStorageBuffer64);
            spvb_extension(emitter->file_builder, "SPV_KHR_physical_storage_buffer");
            spvb_capability(emitter->file_builder, SpvCapabilityPhysicalStorageBufferAddresses);
            return SpvStorageClassPhysicalStorageBuffer;
        case AsInput:
        case AsUInput:                       return SpvStorageClassInput;
        case AsOutput:                       return SpvStorageClassOutput;
        case AsPushConstant:                 return SpvStorageClassPushConstant;
        case AsShaderStorageBufferObject:    return SpvStorageClassStorageBuffer;
        case AsUniform:                      return SpvStorageClassUniform;
        case AsImage:                        return SpvStorageClassImage;
        case AsUniformConstant:              return SpvStorageClassUniformConstant;

        default: {
            shd_error_print("Cannot emit address space %s.\n", shd_get_address_space_name(address_space));
            shd_error_die();
            SHADY_UNREACHABLE;
        }
    }
}

static const Node* rewrite_normalize(Rewriter* rewriter, const Node* node) {
    if (!is_type(node)) {
        shd_register_processed(rewriter, node, node);
        return node;
    }

    switch (node->tag) {
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (QualifiedType) { .type = shd_rewrite_node(rewriter, node->payload.qualified_type.type), .is_uniform = false });
        default: return shd_recreate_node(rewriter, node);
    }
}

const Type* spv_normalize_type(Emitter* emitter, const Type* type) {
    Rewriter rewriter = shd_create_node_rewriter(emitter->module, emitter->module, rewrite_normalize);
    const Node* rewritten = shd_rewrite_node(&rewriter, type);
    shd_destroy_rewriter(&rewriter);
    return rewritten;
}

SpvId spv_types_to_codom(Emitter* emitter, Nodes return_types) {
    switch (return_types.count) {
        case 0: return emitter->void_t;
        case 1: return spv_emit_type(emitter, return_types.nodes[0]);
        default: {
            const Type* codom_ret_type = record_type(emitter->arena, (RecordType) { .members = return_types, .special = 0 });
            return spv_emit_type(emitter, codom_ret_type);
        }
    }
}

void spv_emit_nominal_type_body(Emitter* emitter, const Type* type, SpvId id) {
    switch (type->tag) {
        case RecordType_TAG: {
            Nodes member_types = type->payload.record_type.members;
            LARRAY(SpvId, members, member_types.count);
            for (size_t i = 0; i < member_types.count; i++)
                members[i] = spv_emit_type(emitter, member_types.nodes[i]);
            spvb_struct_type(emitter->file_builder, id, member_types.count, members);
            if (type->payload.record_type.special == DecorateBlock) {
                spvb_decorate(emitter->file_builder, id, SpvDecorationBlock, 0, NULL);
            }
            LARRAY(FieldLayout, fields, member_types.count);
            shd_get_record_layout(emitter->arena, type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                spvb_decorate_member(emitter->file_builder, id, i, SpvDecorationOffset, 1, (uint32_t[]) { fields[i].offset_in_bytes });
            }
            break;
        }
        default: shd_error("not a suitable nominal type body (tag=%s)", shd_get_node_tag_string(type->tag));
    }
}

SpvId spv_emit_type(Emitter* emitter, const Type* type) {
    // Some types in shady lower to the same spir-v type, but spir-v is unhappy with having duplicates of the same types
    // we could hash the spirv types we generate to find duplicates, but it is easier to normalise our shady types and reuse their infra
    type = spv_normalize_type(emitter, type);

    SpvId* existing = spv_search_emitted(emitter, NULL, type);
    if (existing)
        return *existing;

    SpvId new;
    switch (is_type(type)) {
        case NotAType: shd_error("Not a type");
        case Int_TAG: {
            int width;
            switch (type->payload.int_type.width) {
                case IntTy8:
                    spvb_capability(emitter->file_builder, SpvCapabilityInt8);
                    width = 8;  break;
                case IntTy16:
                    spvb_capability(emitter->file_builder, SpvCapabilityInt16);
                    width = 16; break;
                case IntTy32: width = 32; break;
                case IntTy64:
                    spvb_capability(emitter->file_builder, SpvCapabilityInt64);
                    width = 64; break;
                default: assert(false);
            }
            new = spvb_int_type(emitter->file_builder, width, type->payload.int_type.is_signed);
            break;
        } case Bool_TAG: {
            new = spvb_bool_type(emitter->file_builder);
            break;
        } case Float_TAG: {
            int width;
            switch (type->payload.float_type.width) {
                case FloatTy16:
                    spvb_capability(emitter->file_builder, SpvCapabilityFloat16);
                    width = 16; break;
                case FloatTy32: width = 32; break;
                case FloatTy64:
                    spvb_capability(emitter->file_builder, SpvCapabilityFloat64);
                    width = 64; break;
            }
            new = spvb_float_type(emitter->file_builder, width);
            break;
        } case PtrType_TAG: {
            SpvStorageClass sc = spv_emit_addr_space(emitter, type->payload.ptr_type.address_space);
            const Type* pointed_type = type->payload.ptr_type.pointed_type;
            if (get_maybe_nominal_type_decl(pointed_type) && sc == SpvStorageClassPhysicalStorageBuffer) {
                new = spvb_forward_ptr_type(emitter->file_builder, sc);
                spv_register_emitted(emitter, NULL, type, new);
                SpvId pointee = spv_emit_type(emitter, pointed_type);
                spvb_ptr_type_define(emitter->file_builder, new, sc, pointee);
                return new;
            }

            SpvId pointee = spv_emit_type(emitter, pointed_type);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);

            //if (is_physical_as(type->payload.ptr_type.address_space) && type->payload.ptr_type.pointed_type->tag == ArrType_TAG) {
            //    TypeMemLayout elem_mem_layout = get_mem_layout(emitter->arena, type->payload.ptr_type.pointed_type);
            //    spvb_decorate(emitter->file_builder, new, SpvDecorationArrayStride, 1, (uint32_t[]) {elem_mem_layout.size_in_bytes});
            //}
            break;
        }
        case NoRet_TAG:
        case LamType_TAG:
        case BBType_TAG: shd_error("we can't emit arrow types that aren't those of shd_first-class functions")
        case FnType_TAG: {
            const FnType* fnt = &type->payload.fn_type;
            LARRAY(SpvId, params, fnt->param_types.count);
            for (size_t i = 0; i < fnt->param_types.count; i++)
                params[i] = spv_emit_type(emitter, fnt->param_types.nodes[i]);

            new = spvb_fn_type(emitter->file_builder, fnt->param_types.count, params, spv_types_to_codom(emitter, fnt->return_types));
            break;
        }
        case QualifiedType_TAG: {
            // SPIR-V does not care about our type qualifiers.
            new = spv_emit_type(emitter, type->payload.qualified_type.type);
            break;
        }
        case ArrType_TAG: {
            SpvId element_type = spv_emit_type(emitter, type->payload.arr_type.element_type);
            if (type->payload.arr_type.size) {
                new = spvb_array_type(emitter->file_builder, element_type, spv_emit_value(emitter, NULL, type->payload.arr_type.size));
            } else {
                new = spvb_runtime_array_type(emitter->file_builder, element_type);
            }
            TypeMemLayout elem_mem_layout = shd_get_mem_layout(emitter->arena, type->payload.arr_type.element_type);
            spvb_decorate(emitter->file_builder, new, SpvDecorationArrayStride, 1, (uint32_t[]) { elem_mem_layout.size_in_bytes });
            break;
        }
        case PackType_TAG: {
            assert(type->payload.pack_type.width >= 2);
            SpvId element_type = spv_emit_type(emitter, type->payload.pack_type.element_type);
            new = spvb_vector_type(emitter->file_builder, element_type, type->payload.pack_type.width);
            break;
        }
        case RecordType_TAG: {
            if (type->payload.record_type.members.count == 0) {
                new = emitter->void_t;
                break;
            }
            new = spvb_fresh_id(emitter->file_builder);
            spv_register_emitted(emitter, NULL, type, new);
            spv_emit_nominal_type_body(emitter, type, new);
            return new;
        }
        case Type_TypeDeclRef_TAG: {
            new = spv_emit_decl(emitter, type->payload.type_decl_ref.decl);
            break;
        }
        case Type_SampledImageType_TAG: new = spvb_sampled_image_type(emitter->file_builder, spv_emit_type(emitter, type->payload.sampled_image_type.image_type)); break;
        case Type_SamplerType_TAG: new = spvb_sampler_type(emitter->file_builder); break;
        case Type_ImageType_TAG: {
            ImageType p = type->payload.image_type;
            new = spvb_image_type(emitter->file_builder, spv_emit_type(emitter, p.sampled_type), p.dim, p.depth, p.arrayed, p.ms, p.sampled, p.imageformat);
            break;
        }
        case Type_MaskType_TAG:
        case Type_JoinPointType_TAG: shd_error("These must be lowered beforehand")
    }

    if (shd_is_data_type(type)) {
        if (type->tag == PtrType_TAG && type->payload.ptr_type.address_space == AsGlobal) {
            //TypeMemLayout elem_mem_layout = get_mem_layout(emitter->arena, type->payload.ptr_type.pointed_type);
            //spvb_decorate(emitter->file_builder, new, SpvDecorationArrayStride, 1, (uint32_t[]) {elem_mem_layout.size_in_bytes});
        }
    }

    spv_register_emitted(emitter, NULL, type, new);
    return new;
}
