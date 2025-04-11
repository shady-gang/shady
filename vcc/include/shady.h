#ifndef _SHADY_H
#define _SHADY_H

#ifndef __SHADY__
#error "This header can only be used with Vcc"
#endif

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
namespace vcc {
#endif

#define vertex_shader __attribute__((annotate("shady::entry_point::Vertex")))
#define fragment_shader __attribute__((annotate("shady::entry_point::Fragment")))
#define compute_shader  __attribute__((annotate("shady::entry_point::Compute")))
#define ray_generation_shader  __attribute__((annotate("shady::entry_point::RayGeneration")))

#define location(i)            __attribute__((annotate("shady::location::"#i)))
#define descriptor_set(i)      __attribute__((annotate("shady::descriptor_set::"#i)))
#define descriptor_binding(i)  __attribute__((annotate("shady::descriptor_binding::"#i)))
#define local_size(x, y, z)    __attribute__((annotate("shady::workgroup_size::"#x"::"#y"::"#z)))

#define input                  __attribute__((annotate("shady::io::389")))
#define output                 __attribute__((annotate("shady::io::390")))
// maybe deprecate it ?
#define uniform_constant       __attribute__((annotate("shady::io::398")))
#define uniform_block          __attribute__((annotate("shady::io::395")))
#define push_constant          __attribute__((annotate("shady::io::392")))
#define global                 __attribute__((address_space(1)))
#define shared                 __attribute__((address_space(3)))
#define private                __attribute__((address_space(5)))

float sqrtf(float f) __asm__("shady::prim_op::sqrt");

typedef float native_vec4     __attribute__((ext_vector_type(4)));
typedef float native_vec3     __attribute__((ext_vector_type(3)));
typedef float native_vec2     __attribute__((ext_vector_type(2)));

typedef int native_ivec4      __attribute__((ext_vector_type(4)));
typedef int native_ivec3      __attribute__((ext_vector_type(3)));
typedef int native_ivec2      __attribute__((ext_vector_type(2)));

typedef unsigned native_uvec4 __attribute__((ext_vector_type(4)));
typedef unsigned native_uvec3 __attribute__((ext_vector_type(3)));
typedef unsigned native_uvec2 __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(0x1000))) struct __shady_builtin_sampler1D* sampler1D;
typedef __attribute__((address_space(0x1001))) struct __shady_builtin_sampler2D* sampler2D;
typedef __attribute__((address_space(0x1002))) struct __shady_builtin_sampler3D* sampler3D;
typedef __attribute__((address_space(0x1003))) struct __shady_builtin_sampler3D* samplerCube;

native_vec4 texture1D(const sampler1D, float)           __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 texture2D(const sampler2D, native_vec2)     __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 texture3D(const sampler3D, native_vec3)     __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 textureCube(const samplerCube, native_vec3) __asm__("shady::impure_op::spirv.core::87::Invocation");

#if defined(__cplusplus)
native_vec4 texture(const sampler1D, float)         __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 texture(const sampler2D, native_vec2)   __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 texture(const sampler3D, native_vec3)   __asm__("shady::impure_op::spirv.core::87::Invocation");
native_vec4 texture(const samplerCube, native_vec3) __asm__("shady::impure_op::spirv.core::87::Invocation");
#endif

// builtins
__attribute__((annotate("shady::builtin::FragCoord")))
__attribute__((annotate("shady::io::389")))
input native_vec4 gl_FragCoord;

__attribute__((annotate("shady::builtin::Position")))
__attribute__((annotate("shady::io::389")))
output native_vec4 gl_Position;

__attribute__((annotate("shady::builtin::WorkgroupId")))
__attribute__((annotate("shady::io::389")))
native_uvec3 gl_WorkGroupID;

__attribute__((annotate("shady::builtin::VertexIndex")))
__attribute__((annotate("shady::io::389")))
unsigned gl_VertexIndex;

__attribute__((annotate("shady::builtin::SubgroupId")))
__attribute__((annotate("shady::io::389")))
unsigned subgroup_id;

__attribute__((annotate("shady::builtin::SubgroupLocalInvocationId")))
__attribute__((annotate("shady::io::389")))
unsigned subgroup_local_id;

__attribute__((annotate("shady::builtin::WorkgroupSize")))
__attribute__((annotate("shady::io::389")))
native_uvec3 gl_WorkGroupSize;

__attribute__((annotate("shady::builtin::GlobalInvocationId")))
__attribute__((annotate("shady::io::389")))
native_uvec3 gl_GlobalInvocationID;

__attribute__((annotate("shady::builtin::LaunchIdKHR")))
__attribute__((annotate("shady::io::389")))
native_uvec3 gl_LaunchIDEXT;

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
}
#endif

#endif
