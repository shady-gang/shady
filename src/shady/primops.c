#include "shady/ir.h"

#include "portability.h"
#include "log.h"
#include <assert.h>

#include "primops_generated.c"

String shd_get_primop_name(Op op) {
    return primop_names[op];
}

#include "spirv/unified1/GLSL.std.450.h"

static inline const Type* _shd_match_data_types(const Node* a, const Node* b) {
    shd_deconstruct_qualified_type(&a);
    shd_deconstruct_qualified_type(&b);
    assert(a == b);
    return a;
}

const Node* shd_op_fma(IrArena* arena, const Node* a, const Node* b, const Node* c) { return ext_value_helper(arena, _shd_match_data_types(a, b), "GLSL.std.450", GLSLstd450Fma, mk_nodes(arena, a, b, c)); }
const Node* shd_op_fabs(IrArena* arena, const Node* a) { return ext_value_helper(arena, shd_get_unqualified_type(a), "GLSL.std.450", GLSLstd450FAbs, mk_nodes(arena, a)); }
const Node* shd_op_floor(IrArena* arena, const Node* a) { return ext_value_helper(arena, shd_get_unqualified_type(a), "GLSL.std.450", GLSLstd450Floor, mk_nodes(arena, a)); }
const Node* shd_op_ceil(IrArena* arena, const Node* a) { return ext_value_helper(arena, shd_get_unqualified_type(a), "GLSL.std.450", GLSLstd450Ceil, mk_nodes(arena, a)); }

const Node* shd_op_umax(IrArena* arena, const Node* a, const Node* b) { return ext_value_helper(arena, _shd_match_data_types(a, b), "GLSL.std.450", GLSLstd450UMax, mk_nodes(arena, a, b)); }
const Node* shd_op_umin(IrArena* arena, const Node* a, const Node* b) { return ext_value_helper(arena, _shd_match_data_types(a, b), "GLSL.std.450", GLSLstd450UMin, mk_nodes(arena, a, b)); }
const Node* shd_op_smax(IrArena* arena, const Node* a, const Node* b) { return ext_value_helper(arena, _shd_match_data_types(a, b), "GLSL.std.450", GLSLstd450SMax, mk_nodes(arena, a, b)); }
const Node* shd_op_smin(IrArena* arena, const Node* a, const Node* b) { return ext_value_helper(arena, _shd_match_data_types(a, b), "GLSL.std.450", GLSLstd450SMin, mk_nodes(arena, a, b)); }