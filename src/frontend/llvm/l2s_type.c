#include "l2s_private.h"

#include "shady/be/dump.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "util.h"

const Type* l2s_convert_type(Parser* p, LLVMTypeRef t) {
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
                cparam_types[i] = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, l2s_convert_type(p, param_types[i]));
            const Type* ret_type = l2s_convert_type(p, LLVMGetReturnType(t));
            if (LLVMGetTypeKind(LLVMGetReturnType(t)) == LLVMVoidTypeKind)
                ret_type = empty_multiple_return_type(a);
            else
                ret_type = qualified_type_helper(a, shd_get_arena_config(a)->target.scopes.bottom, ret_type);
            return fn_type(a, (FnType) {
                .param_types = shd_nodes(a, num_params, cparam_types),
                .return_types = ret_type == empty_multiple_return_type(a) ? shd_empty(a) : shd_singleton(ret_type)
            });
        }
        case LLVMStructTypeKind: {
            String name = LLVMGetStructName(t);
            Node* struct_t = struct_type_helper(a, 0);
            if (name && strlen(name) > 0)
                shd_set_debug_name(struct_t, name);
            shd_dict_insert(LLVMTypeRef, const Type*, p->map, t, struct_t);

            unsigned size = LLVMCountStructElementTypes(t);
            LARRAY(LLVMTypeRef, elements, size);
            LLVMGetStructElementTypes(t, elements);
            LARRAY(const Type*, celements, size);
            for (size_t i = 0; i < size; i++) {
                celements[i] = l2s_convert_type(p, elements[i]);
            }

            shd_struct_type_set_members(struct_t, shd_nodes(a, size, celements));

            return struct_t;
        }
        case LLVMArrayTypeKind: {
            unsigned length = LLVMGetArrayLength(t);
            const Type* elem_t = l2s_convert_type(p, LLVMGetElementType(t));
            if (!shd_is_physical_data_type(elem_t) && length == 0)
                return arr_type(a, (ArrType) { .element_type = elem_t });
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
                    case 0x0: sampled_type = float_type(a, (Float) {.width = ShdFloatFormat32}); break;
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
            AddressSpace as = l2s_convert_llvm_address_space(llvm_as);
            const Type* pointee = NULL;
#if !UNTYPED_POINTERS
            LLVMTypeRef element_type = LLVMGetElementType(t);
            pointee = l2s_convert_type(p, element_type);
#endif
            if (!pointee || pointee == unit_type(a))
                pointee = shd_uword_type(a);

            return ptr_type(a, (PtrType) {
                .address_space = as,
                .pointed_type = pointee
            });
        }
        case LLVMVectorTypeKind: {
            unsigned width = LLVMGetVectorSize(t);
            const Type* elem_t = l2s_convert_type(p, LLVMGetElementType(t));
            return vector_type(a, (VectorType) { .element_type = elem_t, .width = (size_t) width });
        }
        case LLVMMetadataTypeKind:
            assert(false && "why are we typing metadata");
            break;
        case LLVMTokenTypeKind:
            break;
        case LLVMScalableVectorTypeKind:
        case LLVMBFloatTypeKind:
#if LLVM_VERSION_MAJOR < 20
        case LLVMX86_MMXTypeKind:
            break;
#endif
        case LLVMX86_AMXTypeKind:
            break;
        case LLVMPPC_FP128TypeKind:
            break;
    }

    shd_error_print("Unsupported type: ");
    LLVMDumpType(t);
    shd_error_die();
}

const Type* l2s_get_param_byval_attr(Parser* p, LLVMValueRef fn, size_t param_index) {
    size_t num_attrs = LLVMGetAttributeCountAtIndex(fn, param_index + 1);
    LARRAY(LLVMAttributeRef, attrs, num_attrs);
    LLVMGetAttributesAtIndex(fn, param_index + 1, attrs);
    bool is_byval = false;
    for (size_t i = 0; i < num_attrs; i++) {
        LLVMAttributeRef attr = attrs[i];
        size_t k = LLVMGetEnumAttributeKind(attr);
        size_t e = LLVMGetEnumAttributeKindForName("byval", 5);
        uint64_t value = LLVMGetEnumAttributeValue(attr);
        if (k == e) {
            //printf("p = %zu, i = %zu, k = %zu, e = %zu\n", param_index, i, k, e);
            // printf("p = %zu, value = %zu\n", param_index, value);
            LLVMTypeRef tr = (LLVMTypeRef) value;
            //LLVMDumpType(tr);
            const Type* t = l2s_convert_type(p, tr);
            // shd_dump(t);
            return t;
        }
    }
    return NULL;
}
