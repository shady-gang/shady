#ifndef SHADY_EMIT_BUILTINS
#define SHADY_EMIT_BUILTINS

#include "shady/ir.h"

#define u32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = uint32_type(arena) })
#define i32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = int32_type(arena) })
#define i32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = int32_type(arena) })

#define f32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = fp32_type(arena) })

#define SHADY_BUILTINS() \
BUILTIN(BaseInstance,              AsInput,  uint32_type(arena)   )\
BUILTIN(BaseVertex,                AsInput,  uint32_type(arena)   )\
BUILTIN(DeviceIndex,               AsInput,  uint32_type(arena)   )\
BUILTIN(DrawIndex,                 AsInput,  uint32_type(arena)   )\
BUILTIN(VertexIndex,               AsInput,  uint32_type(arena)   )\
BUILTIN(FragCoord,                 AsInput,  f32vec4_type(arena) )\
BUILTIN(FragDepth,                 AsOutput, fp32_type(arena)    )\
BUILTIN(InstanceId,                AsInput,  uint32_type(arena)   )\
BUILTIN(InvocationId,              AsInput,  uint32_type(arena)   )\
BUILTIN(InstanceIndex,             AsInput,  uint32_type(arena)   )\
BUILTIN(LocalInvocationId,         AsInput,  u32vec3_type(arena) )\
BUILTIN(LocalInvocationIndex,      AsInput,  uint32_type(arena)   )\
BUILTIN(GlobalInvocationId,        AsInput,  u32vec3_type(arena) )\
BUILTIN(WorkgroupId,               AsUInput, u32vec3_type(arena) )\
BUILTIN(WorkgroupSize,             AsUInput, u32vec3_type(arena) )\
BUILTIN(NumSubgroups,              AsUInput, uint32_type(arena)   )\
BUILTIN(NumWorkgroups,             AsUInput, u32vec3_type(arena) )\
BUILTIN(Position,                  AsOutput, f32vec4_type(arena) )\
BUILTIN(PrimitiveId,               AsInput,  uint32_type(arena)   )\
BUILTIN(SubgroupLocalInvocationId, AsInput,  uint32_type(arena)  )\
BUILTIN(SubgroupId,                AsUInput, uint32_type(arena)  )\
BUILTIN(SubgroupSize,              AsInput,  uint32_type(arena)  )\

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

#endif
