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
        case AsIncomingCallableDataKHR:      return SpvStorageClassIncomingCallableDataKHR;
        case AsCallableDataKHR:              return SpvStorageClassCallableDataKHR;

        case AsCode: return SpvStorageClassCodeSectionSHADY;

        default: {
            shd_error_print("Cannot emit address space %s.\n", shd_get_address_space_name(address_space));
            shd_error_die();
            SHADY_UNREACHABLE;
        }
    }
}

const Type* spv_normalize_type(Emitter* emitter, const Type* type) {
    const Node* rewritten = shd_rewrite_node(emitter->normalizer, type);
    return rewritten;
}

SpvId spv_types_to_codom(Emitter* emitter, Nodes return_types) {
    return spv_emit_type(emitter, shd_maybe_multiple_return(emitter->arena, return_types));
    switch (return_types.count) {
        case 0: return emitter->void_t;
        case 1: return spv_emit_type(emitter, return_types.nodes[0]);
        default: {
            IrArena* a = emitter->arena;
            LARRAY(const Type*, stripped, return_types.count);
            for (size_t i = 0; i < return_types.count; i++) {
                stripped[i] = return_types.nodes[i];
                if (stripped[i]->tag == QualifiedType_TAG)
                    stripped[i] = shd_get_unqualified_type(stripped[i]);
            }
            const Type* codom_ret_type = tuple_type(emitter->arena, (TupleType) { .members = shd_nodes(a, return_types.count, stripped) });
            return spv_emit_type(emitter, codom_ret_type);
        }
    }
}

static void spv_emit_type_layout(Emitter* emitter, const Type* type, SpvId id) {
    if (shd_node_set_find(emitter->types_with_layouts, type))
        return;
    shd_node_set_insert(emitter->types_with_layouts, type);
    type = shd_get_maybe_nominal_type_body(type);
    switch (type->tag) {
        case RecordType_TAG: {
            RecordType payload = type->payload.record_type;
            Nodes member_types = payload.members;
            LARRAY(FieldLayout, fields, member_types.count);
            shd_get_record_layout(emitter->arena, type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                spvb_decorate_member(emitter->file_builder, id, i, SpvDecorationOffset, 1, (uint32_t[]) { fields[i].offset_in_bytes });
            }
            break;
        }
        case ArrType_TAG: {
            TypeMemLayout elem_mem_layout = shd_get_mem_layout(emitter->arena, type->payload.arr_type.element_type);
            spvb_decorate(emitter->file_builder, id, SpvDecorationArrayStride, 1, (uint32_t[]) { elem_mem_layout.size_in_bytes });
            break;
        }
        case NominalType_TAG: assert(false);
        case PtrType_TAG: {
            PtrType payload = type->payload.ptr_type;
            if (emitter->target->memory.address_spaces[payload.address_space].physical) {
                TypeMemLayout elem_mem_layout = shd_get_mem_layout(emitter->arena, shd_get_pointer_type_element(type));
                if (elem_mem_layout.size_in_bytes > 0)
                    spvb_decorate(emitter->file_builder, id, SpvDecorationArrayStride, 1, (uint32_t[]) { elem_mem_layout.size_in_bytes });
            }
            break;
        }
        default: break;
    }
}

void spv_emit_record_type_body(Emitter* emitter, const Type* type, SpvId id) {
    if (type->tag != RecordType_TAG)
        shd_error("not a suitable nominal type body (tag=%s)", shd_get_node_tag_string(type->tag));
    RecordType payload = type->payload.record_type;
    Nodes member_types = payload.members;
    LARRAY(SpvId, members, member_types.count);
    for (size_t i = 0; i < member_types.count; i++)
        members[i] = spv_emit_type(emitter, member_types.nodes[i]);
    spvb_struct_type(emitter->file_builder, id, member_types.count, members);

    if (payload.special == ShdRecordFlagBlock) {
        spvb_decorate(emitter->file_builder, id, SpvDecorationBlock, 0, NULL);
    }
}

SpvId spv_emit_type(Emitter* emitter, const Type* type) {
    // Some types in shady lower to the same spir-v type, but spir-v is unhappy with having duplicates of the same types
    // we could hash the spirv types we generate to find duplicates, but it is easier to normalise our shady types and reuse their infra

    // Only non-aggregate, non-pointer types require being unique, and we can just emit all QualifiedTypes their bodies
    // This is only an issue for function types, which need to have
    const Type* key = type;
    if (type->arena == emitter->arena)
        key = spv_normalize_type(emitter, type);

    SpvId* existing = spv_search_emitted(emitter, NULL, key);
    if (existing)
        return *existing;

    SpvId new;
    switch (is_type(type)) {
        case NotAType: shd_error("Not a type");
        case Int_TAG: {
            int width;
            switch (type->payload.int_type.width) {
                case ShdIntSize8:
                    spvb_capability(emitter->file_builder, SpvCapabilityInt8);
                    width = 8;  break;
                case ShdIntSize16:
                    spvb_capability(emitter->file_builder, SpvCapabilityInt16);
                    width = 16; break;
                case ShdIntSize32: width = 32; break;
                case ShdIntSize64:
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
                case ShdFloatFormat16:
                    spvb_capability(emitter->file_builder, SpvCapabilityFloat16);
                    width = 16; break;
                case ShdFloatFormat32: width = 32; break;
                case ShdFloatFormat64:
                    spvb_capability(emitter->file_builder, SpvCapabilityFloat64);
                    width = 64; break;
            }
            new = spvb_float_type(emitter->file_builder, width);
            break;
        } case PtrType_TAG: {
            PtrType payload = type->payload.ptr_type;
            SpvStorageClass sc = spv_emit_addr_space(emitter, payload.address_space);
            const Type* pointed_type = payload.pointed_type;
            if (pointed_type->tag == NominalType_TAG && sc == SpvStorageClassPhysicalStorageBuffer) {
                new = spvb_forward_ptr_type(emitter->file_builder, sc);
                spv_register_emitted(emitter, NULL, key, new);
                spv_emit_type_layout(emitter, type, new);
                SpvId pointee = spv_emit_type(emitter, pointed_type);
                spvb_ptr_type_define(emitter->file_builder, new, sc, pointee);
                return new;
            }

            SpvId pointee = spv_emit_type(emitter, pointed_type);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);
            spv_register_emitted(emitter, NULL, key, new);
            spv_emit_type_layout(emitter, type, new);
            break;
        }
        case NoRet_TAG:
        case LamType_TAG:
        case BBType_TAG: shd_error("we can't emit arrow types that aren't those of shd_first-class functions")
        case FnType_TAG: {
            FnType payload = type->payload.fn_type;
            LARRAY(SpvId, params, payload.param_types.count);
            for (size_t i = 0; i < payload.param_types.count; i++)
                params[i] = spv_emit_type(emitter, payload.param_types.nodes[i]);

            new = spvb_fn_type(emitter->file_builder, payload.param_types.count, params, spv_types_to_codom(emitter, payload.return_types));
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
            spv_emit_type_layout(emitter, type, new);
            break;
        }
        case VectorType_TAG: {
            assert(type->payload.vector_type.width >= 2);
            SpvId element_type = spv_emit_type(emitter, type->payload.vector_type.element_type);
            new = spvb_vector_type(emitter->file_builder, element_type, type->payload.vector_type.width);
            break;
        }
        case Type_MatrixType_TAG: {
            assert(type->payload.matrix_type.columns >= 2);
            SpvId element_type = spv_emit_type(emitter, type->payload.matrix_type.element_type);
            new = spvb_matrix_type(emitter->file_builder, element_type, type->payload.matrix_type.columns);
            break;
        }
        case RecordType_TAG: {
            new = spvb_fresh_id(emitter->file_builder);
            spv_emit_record_type_body(emitter, type, new);
            spv_emit_type_layout(emitter, type, new);
            return new;
        }
        case TupleType_TAG: {
            TupleType payload = type->payload.tuple_type;
            if (payload.members.count == 0) {
                new = emitter->void_t;
                break;
            }
            new = spvb_fresh_id(emitter->file_builder);
            LARRAY(SpvId, members, payload.members.count);
            for (size_t i = 0; i < payload.members.count; i++)
                members[i] = spv_emit_type(emitter, payload.members.nodes[i]);
            spvb_struct_type(emitter->file_builder, new, payload.members.count, members);
            spv_emit_type_layout(emitter, type, new);
            break;
        }
        case NominalType_TAG: {
            new = spv_emit_decl(emitter, type);
            spv_emit_type_layout(emitter, type, new);
            break;
        }
        case Type_SampledImageType_TAG: new = spvb_sampled_image_type(emitter->file_builder, spv_emit_type(emitter, type->payload.sampled_image_type.image_type)); break;
        case Type_SamplerType_TAG: new = spvb_sampler_type(emitter->file_builder); break;
        case Type_ImageType_TAG: {
            ImageType p = type->payload.image_type;
            new = spvb_image_type(emitter->file_builder, spv_emit_type(emitter, p.sampled_type), p.dim, p.depth, p.arrayed, p.ms, p.sampled, p.imageformat);
            break;
        }
        case Type_JoinPointType_TAG: shd_error("These must be lowered beforehand")
        case Type_ExtType_TAG: {
            ExtType payload = type->payload.ext_type;
            if (strcmp(payload.set, "spirv.core") == 0) {
                LARRAY(SpvId, operands, payload.operands.count);
                for (size_t i = 0; i < payload.operands.count; i++)
                    operands[i] = spv_emit_type(emitter, payload.operands.nodes[i]);
                new = spvb_type(emitter->file_builder, payload.opcode, payload.operands.count, operands);
                break;
            }
            shd_error("TODO: extended types")
        }
    }

    if (shd_is_data_type(type)) {
        if (type->tag == PtrType_TAG && type->payload.ptr_type.address_space == AsGlobal) {
            //TypeMemLayout elem_mem_layout = get_mem_layout(emitter->arena, type->payload.ptr_type.pointed_type);
            //spvb_decorate(emitter->file_builder, new, SpvDecorationArrayStride, 1, (uint32_t[]) {elem_mem_layout.size_in_bytes});
        }
    }

    spv_register_emitted(emitter, NULL, key, new);
    return new;
}
