#ifndef SHADY_IR_BUILTIN_H
#define SHADY_IR_BUILTIN_H

#include "shady/ir/base.h"
#include "shady/ir/enum.h"

#define shd_u32vec3_type(arena) vector_type(arena, (VectorType) { .width = 3, .element_type = shd_uint32_type(arena) })
#define shd_i32vec3_type(arena) vector_type(arena, (VectorType) { .width = 3, .element_type = shd_int32_type(arena) })
#define shd_i32vec4_type(arena) vector_type(arena, (VectorType) { .width = 4, .element_type = shd_int32_type(arena) })

#define shd_f32vec4_type(arena) vector_type(arena, (VectorType) { .width = 4, .element_type = shd_fp32_type(arena) })

#define SHADY_BUILTINS() \
BUILTIN(BaseInstance,                AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(BaseVertex,                  AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(DeviceIndex,                 AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(DrawIndex,                   AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(VertexIndex,                 AsInput,  Invocation, shd_int32_type(arena)   )\
BUILTIN(FragCoord,                   AsInput,  Invocation, shd_f32vec4_type(arena) )\
BUILTIN(FragDepth,                   AsOutput, Invocation, shd_fp32_type(arena)    )\
BUILTIN(InstanceId,                  AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(InvocationId,                AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(InstanceIndex,               AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(LocalInvocationId,           AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(LocalInvocationIndex,        AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(GlobalInvocationId,          AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(LaunchIdKHR,                 AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(LaunchSizeKHR,               AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupId,                 AsUInput, Workgroup,  shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupSize,               AsUInput, Device,     shd_u32vec3_type(arena) )\
BUILTIN(NumSubgroups,                AsUInput, Invocation, shd_uint32_type(arena)  )\
BUILTIN(NumWorkgroups,               AsUInput, Device,     shd_u32vec3_type(arena) )\
BUILTIN(Position,                    AsOutput, Invocation, shd_f32vec4_type(arena) )\
BUILTIN(PrimitiveTriangleIndicesEXT, AsOutput, Invocation, arr_type_helper(arena, 0, shd_u32vec3_type(arena), NULL) )\
BUILTIN(PrimitiveId,                 AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(SubgroupLocalInvocationId,   AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(SubgroupId,                  AsUInput, Subgroup,   shd_uint32_type(arena)  )\
BUILTIN(SubgroupSize,                AsInput,  Device,     shd_uint32_type(arena)  )\

typedef enum {
#define BUILTIN(name, as, scope, datatype) ShdBuiltin##name,
SHADY_BUILTINS()
#undef BUILTIN
  ShdBuiltinsCount
} ShdBuiltin;

AddressSpace shd_get_builtin_address_space(ShdBuiltin builtin);
String shd_get_builtin_name(ShdBuiltin builtin);
ShdScope shd_get_builtin_scope(ShdBuiltin builtin);

const Type* shd_get_builtin_type(IrArena* arena, ShdBuiltin builtin);
ShdBuiltin shd_get_builtin_by_name(String s);

#ifdef spirv_H
ShdBuiltin shd_get_builtin_by_spv_id(SpvBuiltIn id);
#endif

int32_t shd_get_builtin_spv_id(ShdBuiltin builtin);

bool shd_is_builtin_load_op(const Node* n, ShdBuiltin* out);

const Node* shd_get_or_create_builtin(Module* m, ShdBuiltin b);

typedef struct BodyBuilder_ BodyBuilder;
const Node* shd_bld_builtin_load(Module* m, BodyBuilder* bb, ShdBuiltin b);

#endif
