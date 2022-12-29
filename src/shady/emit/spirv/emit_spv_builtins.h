#ifndef SHADY_EMIT_BUILTINS
#define SHADY_EMIT_BUILTINS

#include "shady/ir.h"
#include "spirv/unified1/spirv.h"

#define i32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = int32_type(arena) })
#define i32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = int32_type(arena) })

#define f32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = float_type(arena) })

#define VULKAN_BUILTINS() \
BUILTIN(Input,  BaseInstance,              int32_type(arena)   )\
BUILTIN(Input,  BaseVertex,                int32_type(arena)   )\
BUILTIN(Input,  DeviceIndex,               int32_type(arena)   )\
BUILTIN(Input,  DrawIndex,                 int32_type(arena)   )\
BUILTIN(Input,  FragCoord,                 f32vec4_type(arena) )\
BUILTIN(Output, FragDepth,                 float_type(arena)   )\
BUILTIN(Input,  InstanceId,                int32_type(arena)   )\
BUILTIN(Input,  InvocationId,              int32_type(arena)   )\
BUILTIN(Input,  InstanceIndex,             int32_type(arena)   )\
BUILTIN(Input,  LocalInvocationId,         i32vec3_type(arena) )\
BUILTIN(Input,  LocalInvocationIndex,      int32_type(arena)   )\
BUILTIN(Input,  GlobalInvocationId,        i32vec3_type(arena) )\
BUILTIN(Input,  WorkgroupId,               i32vec3_type(arena) )\
BUILTIN(Input,  WorkgroupSize,             i32vec3_type(arena) )\
BUILTIN(Input,  NumSubgroups,              int32_type(arena)   )\
BUILTIN(Input,  NumWorkgroups,             int32_type(arena)   )\
BUILTIN(Output, Position,                  f32vec4_type(arena) )\
BUILTIN(Input,  PrimitiveId,               int32_type(arena)   )\
BUILTIN(Input,  SubgroupLocalInvocationId, int32_type(arena)   )\
BUILTIN(Input,  SubgroupId,                int32_type(arena)   )\
BUILTIN(Input,  SubgroupSize,              int32_type(arena)   )\

typedef enum {
#define BUILTIN(kind, name, datatype) VulkanBuiltin##name,
VULKAN_BUILTINS()
#undef BUILTIN
  VulkanBuiltinsCount
} VulkanBuiltins;

typedef enum {
    VulkanBuiltinInput,
    VulkanBuiltinOutput,
    VulkanBuiltinConstant
} VulkanBuiltinKind;

extern VulkanBuiltinKind vulkan_builtins_kind[];

const Type* get_vulkan_builtins_type(IrArena* arena, VulkanBuiltins builtin);

#endif
