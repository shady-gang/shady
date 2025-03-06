#include "shady/ir/cast.h"
#include "shady/ir/grammar.h"
#include "shady/ir/type.h"
#include "shady/ir/memory_layout.h"

#include <assert.h>

const Node* shd_bld_reinterpret_cast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return bit_cast_helper(shd_get_bb_arena(bb), dst, src);
}

const Node* shd_bld_conversion(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return prim_op(shd_get_bb_arena(bb), (PrimOp) { .op = convert_op, .operands = shd_singleton(src), .type_arguments = shd_singleton(dst)});
}

bool shd_is_reinterpret_cast_legal(const Type* src_type, const Type* dst_type) {
    assert(shd_is_data_type(src_type) && shd_is_data_type(dst_type));
    if (src_type == dst_type)
        return true; // folding will eliminate those, but we need to pass type-checking first :)
    if (!(shd_is_arithm_type(src_type) || src_type->tag == MaskType_TAG || shd_is_physical_ptr_type(src_type)))
        return false;
    if (!(shd_is_arithm_type(dst_type) || dst_type->tag == MaskType_TAG || shd_is_physical_ptr_type(dst_type)))
        return false;
    assert(shd_get_type_bitwidth(src_type) == shd_get_type_bitwidth(dst_type));
    // either both pointers need to be in the generic address space, and we're only casting the element type, OR neither can be
    if ((shd_is_physical_ptr_type(src_type) && shd_is_physical_ptr_type(dst_type)) && (shd_is_generic_ptr_type(src_type) != shd_is_generic_ptr_type(dst_type)))
        return false;
    return true;
}

bool shd_is_conversion_legal(const Type* src_type, const Type* dst_type) {
    assert(shd_is_data_type(src_type) && shd_is_data_type(dst_type));
    if (!(shd_is_arithm_type(src_type) || (shd_is_physical_ptr_type(src_type) && shd_get_type_bitwidth(src_type) == shd_get_type_bitwidth(dst_type))))
        return false;
    if (!(shd_is_arithm_type(dst_type) || (shd_is_physical_ptr_type(dst_type) && shd_get_type_bitwidth(src_type) == shd_get_type_bitwidth(dst_type))))
        return false;
    // we only allow ptr-ptr conversions, use reinterpret otherwise
    if (shd_is_physical_ptr_type(src_type) != shd_is_physical_ptr_type(dst_type))
        return false;
    // exactly one of the pointers needs to be in the generic address space
    if (shd_is_generic_ptr_type(src_type) && shd_is_generic_ptr_type(dst_type))
        return false;
    if (src_type->tag == Int_TAG && dst_type->tag == Int_TAG) {
        bool changes_sign = src_type->payload.int_type.is_signed != dst_type->payload.int_type.is_signed;
        bool changes_width = src_type->payload.int_type.width != dst_type->payload.int_type.width;
        if (changes_sign && changes_width)
            return false;
    }
    // element types have to match (use reinterpret_cast for changing it)
    if (shd_is_physical_ptr_type(src_type) && shd_is_physical_ptr_type(dst_type)) {
        AddressSpace src_as = src_type->payload.ptr_type.address_space;
        AddressSpace dst_as = dst_type->payload.ptr_type.address_space;
        if (src_type->payload.ptr_type.pointed_type != dst_type->payload.ptr_type.pointed_type)
            return false;
    }
    return true;
}
