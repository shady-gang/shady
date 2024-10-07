#include "l2s_private.h"
#include "type.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "util.h"

const Type* convert_type(Parser* p, LLVMTypeRef t) {
    const Type** found = shd_dict_find_value(LLVMTypeRef, const Type*, p->map, t);
    if (found) return *found;
    IrArena* a = shd_module_get_arena(p->dst);

    switch (LLVMGetTypeKind(t)) {
        case LLVMVoidTypeKind: return unit_type(a);
        case LLVMHalfTypeKind: return shd_fp16_type(a);
        case LLVMFloatTypeKind: return shd_fp32_type(a);
        case LLVMDoubleTypeKind: return shd_fp64_type(a);
        case LLVMX86_FP80TypeKind:
        case LLVMFP128TypeKind:
            break;
        case LLVMLabelTypeKind:
            break;
        case LLVMIntegerTypeKind:
            switch(LLVMGetIntTypeWidth(t)) {
                case 1: return bool_type(a);
                case 8: return shd_uint8_type(a);
                case 16: return shd_uint16_type(a);
                case 32: return shd_uint32_type(a);
                case 64: return shd_uint64_type(a);
                default: shd_error("Unsupported integer width: %d\n", LLVMGetIntTypeWidth(t)); break;
            }
        case LLVMFunctionTypeKind: {
            unsigned num_params = LLVMCountParamTypes(t);
            LARRAY(LLVMTypeRef, param_types, num_params);
            LLVMGetParamTypes(t, param_types);
            LARRAY(const Type*, cparam_types, num_params);
            for (size_t i = 0; i < num_params; i++)
                cparam_types[i] = shd_as_qualified_type(convert_type(p, param_types[i]), false);
            const Type* ret_type = convert_type(p, LLVMGetReturnType(t));
            if (LLVMGetTypeKind(LLVMGetReturnType(t)) == LLVMVoidTypeKind)
                ret_type = empty_multiple_return_type(a);
            else
                ret_type = shd_as_qualified_type(ret_type, false);
            return fn_type(a, (FnType) {
                .param_types = shd_nodes(a, num_params, cparam_types),
                .return_types = ret_type == empty_multiple_return_type(a) ? shd_empty(a) : shd_singleton(ret_type)
            });
        }
        case LLVMStructTypeKind: {
            String name = LLVMGetStructName(t);
            Node* decl = NULL;
            const Node* result = NULL;
            if (name) {
                decl = nominal_type(p->dst, shd_empty(a), name);
                result = type_decl_ref_helper(a, decl);
                shd_dict_insert(LLVMTypeRef, const Type*, p->map, t, result);
            }

            unsigned size = LLVMCountStructElementTypes(t);
            LARRAY(LLVMTypeRef, elements, size);
            LLVMGetStructElementTypes(t, elements);
            LARRAY(const Type*, celements, size);
            for (size_t i = 0; i < size; i++) {
                celements[i] = convert_type(p, elements[i]);
            }

            const Node* product = record_type(a, (RecordType) {
                .members = shd_nodes(a, size, celements)
            });
            if (decl)
                decl->payload.nom_type.body = product;
            else
                result = product;
            return result;
        }
        case LLVMArrayTypeKind: {
            unsigned length = LLVMGetArrayLength(t);
            const Type* elem_t = convert_type(p, LLVMGetElementType(t));
            return arr_type(a, (ArrType) { .element_type = elem_t, .size = shd_uint32_literal(a, length)});
        }
        case LLVMPointerTypeKind: {
            unsigned int llvm_as = LLVMGetPointerAddressSpace(t);
            if (llvm_as >= 0x1000 && llvm_as <= 0x2000) {
                unsigned offset = llvm_as - 0x1000;
                unsigned dim = offset & 0xF;
                unsigned type_id = (offset >> 4) & 0x3;
                const Type* sampled_type = NULL;
                switch (type_id) {
                    case 0x0: sampled_type = float_type(a, (Float) {.width = FloatTy32}); break;
                    case 0x1: sampled_type = shd_int32_type(a); break;
                    case 0x2: sampled_type = shd_uint32_type(a); break;
                    default: assert(false);
                }
                bool arrayed = (offset >> 6) & 1;

                return sampled_image_type(a, (SampledImageType) {.image_type = image_type(a, (ImageType) {
                        //.sampled_type = pack_type(a, (PackType) { .element_type = float_type(a, (Float) { .width = FloatTy32 }), .width = 4 }),
                        .sampled_type = sampled_type,
                        .dim = dim,
                        .depth = 0,
                        .arrayed = arrayed,
                        .ms = 0,
                        .sampled = 1,
                        .imageformat = 0
                })});
            }
            AddressSpace as = convert_llvm_address_space(llvm_as);
            const Type* pointee = NULL;
#if !UNTYPED_POINTERS
            LLVMTypeRef element_type = LLVMGetElementType(t);
            pointee = convert_type(p, element_type);
#else
            pointee = unit_type(a);
#endif
            return ptr_type(a, (PtrType) {
                .address_space = as,
                .pointed_type = pointee
            });
        }
        case LLVMVectorTypeKind: {
            unsigned width = LLVMGetVectorSize(t);
            const Type* elem_t = convert_type(p, LLVMGetElementType(t));
            return pack_type(a, (PackType) { .element_type = elem_t, .width = (size_t) width });
        }
        case LLVMMetadataTypeKind:
            assert(false && "why are we typing metadata");
            break;
        case LLVMX86_MMXTypeKind:
            break;
        case LLVMTokenTypeKind:
            break;
        case LLVMScalableVectorTypeKind:
        case LLVMBFloatTypeKind:
        case LLVMX86_AMXTypeKind:
            break;
        case LLVMPPC_FP128TypeKind:
            break;
    }

    shd_error_print("Unsupported type: ");
    LLVMDumpType(t);
    shd_error_die();
}
