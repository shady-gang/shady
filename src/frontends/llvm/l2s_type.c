#include "l2s_private.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "util.h"

const Type* convert_type(Parser* p, LLVMTypeRef t) {
    const Type** found = find_value_dict(LLVMTypeRef, const Type*, p->map, t);
    if (found) return *found;
    IrArena* a = get_module_arena(p->dst);

    switch (LLVMGetTypeKind(t)) {
        case LLVMVoidTypeKind: return unit_type(a);
        case LLVMHalfTypeKind: return fp16_type(a);
        case LLVMFloatTypeKind: return fp32_type(a);
        case LLVMDoubleTypeKind: return fp64_type(a);
        case LLVMX86_FP80TypeKind:
        case LLVMFP128TypeKind:
            break;
        case LLVMLabelTypeKind:
            break;
        case LLVMIntegerTypeKind:
            switch(LLVMGetIntTypeWidth(t)) {
                case 1: return bool_type(a);
                case 8: return uint8_type(a);
                case 16: return uint16_type(a);
                case 32: return uint32_type(a);
                case 64: return uint64_type(a);
                default: error("Unsupported integer width: %d\n", LLVMGetIntTypeWidth(t)); break;
            }
        case LLVMFunctionTypeKind: {
            unsigned num_params = LLVMCountParamTypes(t);
            LARRAY(LLVMTypeRef, param_types, num_params);
            LLVMGetParamTypes(t, param_types);
            LARRAY(const Type*, cparam_types, num_params);
            for (size_t i = 0; i < num_params; i++)
                cparam_types[i] = convert_type(p, param_types[i]);
            const Type* ret_type = convert_type(p, LLVMGetReturnType(t));
            return fn_type(a, (FnType) {
                .param_types = nodes(a, num_params, cparam_types),
                .return_types = ret_type == unit_type(a) ? empty(a) : singleton(ret_type)
            });
        }
        case LLVMStructTypeKind: {
            String name = LLVMGetStructName(t);
            Node* decl = NULL;
            const Node* result = NULL;
            if (name) {
                if (strcmp(name, "struct.__shady_builtin_sampler2D") == 0)
                    return combined_image_sampler_type(a, (CombinedImageSamplerType) { .image_type = image_type(a, (ImageType) {
                        //.sampled_type = pack_type(a, (PackType) { .element_type = float_type(a, (Float) { .width = FloatTy32 }), .width = 4 }),
                        .sampled_type = float_type(a, (Float) { .width = FloatTy32 }),
                        .dim = 1,
                        .depth = 0,
                        .onion = 0,
                        .multisample = 0,
                        .sampled = 1,
                    } ) });

                decl = nominal_type(p->dst, empty(a), name);
                result = type_decl_ref_helper(a, decl);
                insert_dict(LLVMTypeRef, const Type*, p->map, t, result);
            }

            unsigned size = LLVMCountStructElementTypes(t);
            LARRAY(LLVMTypeRef, elements, size);
            LLVMGetStructElementTypes(t, elements);
            LARRAY(const Type*, celements, size);
            for (size_t i = 0; i < size; i++) {
                celements[i] = convert_type(p, elements[i]);
            }

            const Node* product = record_type(a, (RecordType) {
                    .members = nodes(a, size, celements)
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
            return arr_type(a, (ArrType) { .element_type = elem_t, .size = uint32_literal(a, length)});
        }
        case LLVMPointerTypeKind: {
            AddressSpace as = convert_llvm_address_space(LLVMGetPointerAddressSpace(t));
            const Type* pointee = uint8_type(a);
#if !UNTYPED_POINTERS
            LLVMTypeRef element_type = LLVMGetElementType(t);
            pointee = convert_type(p, element_type);
            if (LLVMGetTypeKind(element_type) == LLVMStructTypeKind && LLVMGetStructName(element_type) && string_starts_with(LLVMGetStructName(element_type), "struct.__shady_builtin"))
                return pointee;
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

    error_print("Unsupported type: ");
    LLVMDumpType(t);
    error_die();
}
