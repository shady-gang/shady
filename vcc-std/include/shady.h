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

#define location(i)            __attribute__((annotate("shady::location::"#i)))
#define descriptor_set(i)      __attribute__((annotate("shady::descriptor_set::"#i)))
#define descriptor_binding(i)  __attribute__((annotate("shady::descriptor_binding::"#i)))
#define local_size(x, y, z)    __attribute__((annotate("shady::workgroup_size::"#x"::"#y"::"#z)))

#define input                  __attribute__((address_space(389)))
#define output                 __attribute__((address_space(390)))
#define uniform                __attribute__((annotate("shady::uniform")))
#define push_constant          __attribute__((address_space(392)))
#define private                __attribute__((address_space(5)))
#define private_logical        __attribute__((address_space(385)))

typedef float vec4     __attribute__((ext_vector_type(4)));
typedef float vec3     __attribute__((ext_vector_type(3)));
typedef float vec2     __attribute__((ext_vector_type(2)));

typedef int ivec4     __attribute__((ext_vector_type(4)));
typedef int ivec3     __attribute__((ext_vector_type(3)));
typedef int ivec2     __attribute__((ext_vector_type(2)));

typedef unsigned uvec4     __attribute__((ext_vector_type(4)));
typedef unsigned uvec3     __attribute__((ext_vector_type(3)));
typedef unsigned uvec2     __attribute__((ext_vector_type(2)));

typedef struct __shady_builtin_sampler2D {} sampler2D;

vec4 texture2D(const sampler2D, vec2) __asm__("shady::prim_op::sample_texture");

// builtins
__attribute__((annotate("shady::builtin::FragCoord")))
input vec4 gl_FragCoord;

__attribute__((annotate("shady::builtin::Position")))
output vec4 gl_Position;

__attribute__((annotate("shady::builtin::WorkgroupId")))
__attribute__((address_space(389)))
uvec3 gl_WorkGroupID;

__attribute__((annotate("shady::builtin::VertexIndex")))
__attribute__((address_space(389)))
input int gl_VertexIndex;

__attribute__((annotate("shady::builtin::WorkgroupSize")))
__attribute__((address_space(389)))
uvec3 gl_WorkGroupSize;

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
}
#endif

#endif
