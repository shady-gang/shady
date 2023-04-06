#include "emit_spv.h"

#include "portability.h"
#include "log.h"

#include "../../rewrite.h"
#include "../../transform/memory_layout.h"

#include "dict.h"

#include "assert.h"

#pragma GCC diagnostic error "-Wswitch"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

SpvStorageClass emit_addr_space(AddressSpace address_space) {
    switch(address_space) {
        case AsGlobalLogical:                return SpvStorageClassStorageBuffer;
        case AsSharedLogical:                return SpvStorageClassWorkgroup;
        case AsPrivateLogical:               return SpvStorageClassPrivate;
        case AsSPVFunctionLogical:           return SpvStorageClassFunction;
        case AsGlobalPhysical:               return SpvStorageClassPhysicalStorageBuffer;
        case AsInput:                        return SpvStorageClassInput;
        case AsOutput:                       return SpvStorageClassOutput;
        case AsVKPushConstant:               return SpvStorageClassPushConstant;
        case AsGLShaderStorageBufferObject:  return SpvStorageClassStorageBuffer;
        case AsGLUniformBufferObject:        return SpvStorageClassUniform;

        case AsExternal:
        case AsGeneric:
        case AsSubgroupLogical:
        case AsProgramCode:
        case AsSharedPhysical:
        case AsSubgroupPhysical:
        case AsPrivatePhysical: error("This should have been lowered before");
        case NumAddressSpaces: error("These are not usable address spaces")
    }
}

static const Node* rewrite_normalize(Rewriter* rewriter, const Node* node) {
    const Node* found = search_processed(rewriter, node);
    if (found) return found;

    if (!is_type(node)) {
        register_processed(rewriter, node, node);
        return node;
    }

    switch (node->tag) {
        case QualifiedType_TAG: return qualified_type(rewriter->dst_arena, (QualifiedType) { .type = rewrite_node(rewriter, node->payload.qualified_type.type), .is_uniform = false });
        default: return recreate_node_identity(rewriter, node);
    }
}

const Type* normalize_type(Emitter* emitter, const Type* type) {
    Rewriter rewriter = create_rewriter(emitter->module, emitter->module, rewrite_normalize);
    const Node* rewritten = rewrite_node(&rewriter, type);
    destroy_rewriter(&rewriter);
    return rewritten;
}

SpvId nodes_to_codom(Emitter* emitter, Nodes return_types) {
    switch (return_types.count) {
        case 0: return emitter->void_t;
        case 1: return emit_type(emitter, return_types.nodes[0]);
        default: {
            const Type* codom_ret_type = record_type(emitter->arena, (RecordType) { .members = return_types, .special = MultipleReturn });
            return emit_type(emitter, codom_ret_type);
        }
    }
}

void emit_nominal_type_body(Emitter* emitter, const Type* type, SpvId id) {
    switch (type->tag) {
        case RecordType_TAG: {
            Nodes member_types = type->payload.record_type.members;
            LARRAY(SpvId, members, member_types.count);
            for (size_t i = 0; i < member_types.count; i++)
                members[i] = emit_type(emitter, member_types.nodes[i]);
            spvb_struct_type(emitter->file_builder, id, member_types.count, members);
            if (type->payload.record_type.special == DecorateBlock) {
                spvb_decorate(emitter->file_builder, id, SpvDecorationBlock, 0, NULL);
            }
            LARRAY(FieldLayout, fields, member_types.count);
            get_record_layout(emitter->arena, type, fields);
            for (size_t i = 0; i < member_types.count; i++) {
                spvb_decorate_member(emitter->file_builder, id, i, SpvDecorationOffset, 1, (uint32_t[]) { fields[i].offset_in_bytes });
            }
            break;
        }
        default: error("not a suitable nominal type body (tag=%s)", node_tags[type->tag]);
    }
}

SpvId emit_type(Emitter* emitter, const Type* type) {
    // Some types in shady lower to the same spir-v type, but spir-v is unhappy with having duplicates of the same types
    // we could hash the spirv types we generate to find duplicates, but it is easier to normalise our shady types and reuse their infra
    type = normalize_type(emitter, type);

    SpvId* existing = find_value_dict(struct Node*, SpvId, emitter->node_ids, type);
    if (existing)
        return *existing;

    SpvId new;
    switch (is_type(type)) {
        case NotAType: error("Not a type");
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
            SpvId pointee = emit_type(emitter, type->payload.ptr_type.pointed_type);
            SpvStorageClass sc = emit_addr_space(type->payload.ptr_type.address_space);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);
            break;
        }
        case NoRet_TAG:
        case LamType_TAG:
        case BBType_TAG: error("we can't emit arrow types that aren't those of first-class functions")
        case FnType_TAG: {
            const FnType* fnt = &type->payload.fn_type;
            LARRAY(SpvId, params, fnt->param_types.count);
            for (size_t i = 0; i < fnt->param_types.count; i++)
                params[i] = emit_type(emitter, fnt->param_types.nodes[i]);

            new = spvb_fn_type(emitter->file_builder, fnt->param_types.count, params, nodes_to_codom(emitter, fnt->return_types));
            break;
        }
        case QualifiedType_TAG: {
            // SPIR-V does not care about our type qualifiers.
            new = emit_type(emitter, type->payload.qualified_type.type);
            break;
        }
        case ArrType_TAG: {
            SpvId element_type = emit_type(emitter, type->payload.arr_type.element_type);
            if (type->payload.arr_type.size) {
                new = spvb_array_type(emitter->file_builder, element_type, emit_value(emitter, NULL, type->payload.arr_type.size));
            } else {
                new = spvb_runtime_array_type(emitter->file_builder, element_type);
            }
            TypeMemLayout elem_mem_layout = get_mem_layout(emitter->arena, type->payload.arr_type.element_type);
            spvb_decorate(emitter->file_builder, new, SpvDecorationArrayStride, 1, (uint32_t[]) { elem_mem_layout.size_in_bytes });
            break;
        }
        case PackType_TAG: {
            assert(type->payload.pack_type.width >= 2);
            SpvId element_type = emit_type(emitter, type->payload.pack_type.element_type);
            new = spvb_vector_type(emitter->file_builder, element_type, type->payload.pack_type.width);
            break;
        }
        case RecordType_TAG: {
            if (type->payload.record_type.members.count == 0) {
                new = emitter->void_t;
                break;
            }
            new = spvb_fresh_id(emitter->file_builder);
            emit_nominal_type_body(emitter, type, new);
            break;
        }
        case Type_TypeDeclRef_TAG: {
            new = emit_decl(emitter, type->payload.type_decl_ref.decl);
            break;
        }
        case Type_MaskType_TAG:
        case Type_JoinPointType_TAG: error("These must be lowered beforehand")
    }

    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, type, new);
    return new;
}
