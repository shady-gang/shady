#ifndef SHADY_EMIT_BUILTINS
#define SHADY_EMIT_BUILTINS

#include "shady/ir.h"

#define u32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = uint32_type(arena) })
#define i32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = int32_type(arena) })
#define i32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = int32_type(arena) })

#define f32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = fp32_type(arena) })

#define SHADY_BUILTINS() \
BUILTIN(BaseInstance,              AsInput,  int32_type(arena)   )\
BUILTIN(BaseVertex,                AsInput,  int32_type(arena)   )\
BUILTIN(DeviceIndex,               AsInput,  int32_type(arena)   )\
BUILTIN(DrawIndex,                 AsInput,  int32_type(arena)   )\
BUILTIN(VertexIndex,               AsInput,  int32_type(arena)   )\
BUILTIN(FragCoord,                 AsInput,  f32vec4_type(arena) )\
BUILTIN(FragDepth,                 AsOutput, fp32_type(arena)    )\
BUILTIN(InstanceId,                AsInput,  int32_type(arena)   )\
BUILTIN(InvocationId,              AsInput,  int32_type(arena)   )\
BUILTIN(InstanceIndex,             AsInput,  int32_type(arena)   )\
BUILTIN(LocalInvocationId,         AsInput,  i32vec3_type(arena) )\
BUILTIN(LocalInvocationIndex,      AsInput,  int32_type(arena)   )\
BUILTIN(GlobalInvocationId,        AsInput,  u32vec3_type(arena) )\
BUILTIN(WorkgroupId,               AsUInput, i32vec3_type(arena) )\
BUILTIN(WorkgroupSize,             AsUInput, i32vec3_type(arena) )\
BUILTIN(NumSubgroups,              AsUInput, int32_type(arena)   )\
BUILTIN(NumWorkgroups,             AsUInput, int32_type(arena)   )\
BUILTIN(Position,                  AsOutput, f32vec4_type(arena) )\
BUILTIN(PrimitiveId,               AsInput,  int32_type(arena)   )\
BUILTIN(SubgroupLocalInvocationId, AsInput,  uint32_type(arena)  )\
BUILTIN(SubgroupId,                AsUInput, uint32_type(arena)  )\
BUILTIN(SubgroupSize,              AsInput,  uint32_type(arena)  )\

typedef enum {
#define BUILTIN(name, as, datatype) Builtin##name,
SHADY_BUILTINS()
#undef BUILTIN
  BuiltinsCount
} Builtin;

extern AddressSpace builtin_as[];
extern String builtin_names[];

const Type* get_builtin_type(IrArena* arena, Builtin);
Builtin get_builtin_by_name(String);

#endif
