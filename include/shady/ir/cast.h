#ifndef SHADY_IR_CAST_H
#define SHADY_IR_CAST_H

#include "shady/ir/base.h"
#include "shady/ir/builder.h"

const Node* shd_bld_bitcast(BodyBuilder* bb, const Type* dst, const Node* src);
const Node* shd_bld_conversion(BodyBuilder* bb, const Type* dst, const Node* src);

bool shd_is_bitcast_legal(const Type* src_type, const Type* dst_type);
bool shd_is_conversion_legal(const Type* src_type, const Type* dst_type);

#endif
