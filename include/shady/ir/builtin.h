#ifndef SHADY_IR_BUILTIN_H
#define SHADY_IR_BUILTIN_H

#include "shady/ir/base.h"
#include "shady/ir/enum.h"

#define shd_u32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = shd_uint32_type(arena) })
#define shd_i32vec3_type(arena) pack_type(arena, (PackType) { .width = 3, .element_type = shd_int32_type(arena) })
#define shd_i32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = shd_int32_type(arena) })

#define shd_f32vec4_type(arena) pack_type(arena, (PackType) { .width = 4, .element_type = shd_fp32_type(arena) })

#define SHADY_BUILTINS() \
BUILTIN(BaseInstance,              AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(BaseVertex,                AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(DeviceIndex,               AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(DrawIndex,                 AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(VertexIndex,               AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(FragCoord,                 AsInput,  Invocation, shd_f32vec4_type(arena) )\
BUILTIN(FragDepth,                 AsOutput, Invocation, shd_fp32_type(arena)    )\
BUILTIN(InstanceId,                AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(InvocationId,              AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(InstanceIndex,             AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(LocalInvocationId,         AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(LocalInvocationIndex,      AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(GlobalInvocationId,        AsInput,  Invocation, shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupId,               AsUInput, Workgroup,  shd_u32vec3_type(arena) )\
BUILTIN(WorkgroupSize,             AsUInput, Device,     shd_u32vec3_type(arena) )\
BUILTIN(NumSubgroups,              AsUInput, Invocation, shd_uint32_type(arena)   )\
BUILTIN(NumWorkgroups,             AsUInput, Device,     shd_u32vec3_type(arena) )\
BUILTIN(Position,                  AsOutput, Invocation, shd_f32vec4_type(arena) )\
BUILTIN(PrimitiveId,               AsInput,  Invocation, shd_uint32_type(arena)   )\
BUILTIN(SubgroupLocalInvocationId, AsInput,  Invocation, shd_uint32_type(arena)  )\
BUILTIN(SubgroupId,                AsUInput, Subgroup,   shd_uint32_type(arena)  )\
BUILTIN(SubgroupSize,              AsInput,  Device,     shd_uint32_type(arena)  )\

typedef enum {
#define BUILTIN(name, as, scope, datatype) Builtin##name,
SHADY_BUILTINS()
#undef BUILTIN
  BuiltinsCount
} Builtin;

AddressSpace shd_get_builtin_address_space(Builtin builtin);
String shd_get_builtin_name(Builtin builtin);
ShdScope shd_get_builtin_scope(Builtin builtin);

const Type* shd_get_builtin_type(IrArena* arena, Builtin builtin);
Builtin shd_get_builtin_by_name(String s);

#ifdef spirv_H
Builtin shd_get_builtin_by_spv_id(SpvBuiltIn id);
#endif

bool shd_is_decl_builtin(const Node* decl);
Builtin shd_get_decl_builtin(const Node* decl);

int32_t shd_get_builtin_spv_id(Builtin builtin);

bool shd_is_builtin_load_op(const Node* n, Builtin* out);

const Node* shd_get_builtin(Module* m, Builtin b);
const Node* shd_get_or_create_builtin(Module* m, Builtin b, String n);

typedef struct BodyBuilder_ BodyBuilder;
const Node* shd_bld_builtin_load(Module* m, BodyBuilder* bb, Builtin b);

#endif
