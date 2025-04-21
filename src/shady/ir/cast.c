#include "shady/ir/cast.h"
#include "shady/ir/grammar.h"
#include "shady/ir/type.h"
#include "shady/ir/memory_layout.h"

#include <assert.h>

const Node* shd_bld_bitcast(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return bit_cast_helper(shd_get_bb_arena(bb), dst, src);
}

const Node* shd_bld_generic_ptr_cast(BodyBuilder* bb, const Node* src) {
    return generic_ptr_cast_helper(shd_get_bb_arena(bb), src);
}

const Node* shd_bld_conversion(BodyBuilder* bb, const Type* dst, const Node* src) {
    assert(is_type(dst));
    return conversion_helper(shd_get_bb_arena(bb), dst, src);
}

bool shd_is_bitcast_legal(const Type* src_type, const Type* dst_type) {
    assert(shd_is_data_type(src_type) && shd_is_data_type(dst_type));
    if (src_type == dst_type)
        return true; // folding will eliminate those, but we need to pass type-checking first :)
    if (!(shd_is_arithm_type(src_type) || shd_is_physical_ptr_type(src_type)))
        return false;
    if (!(shd_is_arithm_type(dst_type) || shd_is_physical_ptr_type(dst_type)))
        return false;
    assert(shd_get_type_bitwidth(src_type) == shd_get_type_bitwidth(dst_type));
    // either both pointers need to be in the generic address space, and we're only casting the element type, OR neither can be
    if ((shd_is_physical_ptr_type(src_type) && shd_is_physical_ptr_type(dst_type)) && (shd_is_generic_ptr_type(src_type) != shd_is_generic_ptr_type(dst_type)))
        return false;
    return true;
}

bool shd_is_conversion_legal(const Type* src_type, const Type* dst_type) {
    assert(shd_is_data_type(src_type) && shd_is_data_type(dst_type));
    if (!shd_is_arithm_type(src_type))
        return false;
    if (!shd_is_arithm_type(dst_type))
        return false;
    if (src_type->tag == Int_TAG && dst_type->tag == Int_TAG) {
        bool changes_sign = src_type->payload.int_type.is_signed != dst_type->payload.int_type.is_signed;
        bool changes_width = src_type->payload.int_type.width != dst_type->payload.int_type.width;
        if (changes_sign && changes_width)
            return false;
    }
    return true;
}
