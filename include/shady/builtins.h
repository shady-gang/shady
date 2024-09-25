#ifndef SHADY_EMIT_BUILTINS
#define SHADY_EMIT_BUILTINS

#include "shady/ir.h"

#define shd_u32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = shd_uint32_type(arena) })
#define shd_i32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = shd_int32_type(arena) })
#define shd_i32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = shd_int32_type(arena) })

#define shd_f32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = shd_fp32_type(arena) })

#define SHADY_BUILTINS() \
BUILTIN(BaseInstance,              AsInput,  shd_uint32_type(arena)   )\
BUILTIN(BaseVertex,                AsInput,  shd_uint32_type(arena)   )\
BUILTIN(DeviceIndex,               AsInput,  shd_uint32_type(arena)   )\
BUILTIN(DrawIndex,                 AsInput,  shd_uint32_type(arena)   )\
BUILTIN(VertexIndex,               AsInput,  shd_uint32_type(arena)   )\
BUILTIN(FragCoord,                 AsInput,  shd_f32vec4_type(arena) )\
BUILTIN(FragDepth,                 AsOutput, shd_fp32_type(arena)    )\
BUILTIN(InstanceId,                AsInput,  shd_uint32_type(arena)   )\
BUILTIN(InvocationId,              AsInput,  shd_uint32_type(arena)   )\
BUILTIN(InstanceIndex,             AsInput,  shd_uint32_type(arena)   )\
BUILTIN(LocalInvocationId,         AsInput,  shd_u32vec3_type(arena) )\
BUILTIN(LocalInvocationIndex,      AsInput,  shd_uint32_type(arena)   )\
BUILTIN(GlobalInvocationId,        AsInput,  shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupId,               AsUInput, shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupSize,             AsUInput, shd_u32vec3_type(arena) )\
BUILTIN(NumSubgroups,              AsUInput, shd_uint32_type(arena)   )\
BUILTIN(NumWorkgroups,             AsUInput, shd_u32vec3_type(arena) )\
BUILTIN(Position,                  AsOutput, shd_f32vec4_type(arena) )\
BUILTIN(PrimitiveId,               AsInput,  shd_uint32_type(arena)   )\
BUILTIN(SubgroupLocalInvocationId, AsInput,  shd_uint32_type(arena)  )\
BUILTIN(SubgroupId,                AsUInput, shd_uint32_type(arena)  )\
BUILTIN(SubgroupSize,              AsInput,  shd_uint32_type(arena)  )\

typedef enum {
#define BUILTIN(name, as, datatype) Builtin##name,
SHADY_BUILTINS()
#undef BUILTIN
  BuiltinsCount
} Builtin;

AddressSpace get_builtin_as(Builtin);
String get_builtin_name(Builtin);

const Type* get_builtin_type(IrArena* arena, Builtin);
Builtin get_builtin_by_name(String);

typedef enum SpvBuiltIn_ SpvBuiltIn;
Builtin get_builtin_by_spv_id(SpvBuiltIn id);

bool is_decl_builtin(const Node*);
Builtin get_decl_builtin(const Node*);

#endif
