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
#define global                 __attribute__((address_space(1)))
#define private                __attribute__((address_space(5)))
#define private_logical        __attribute__((address_space(385)))

#include "shady_vec.h"

typedef struct __shady_builtin_sampler2D {} sampler2D;

vec4 texture2D(const sampler2D, native_vec2) __asm__("shady::prim_op::sample_texture");

// builtins
__attribute__((annotate("shady::builtin::FragCoord")))
input native_vec4 gl_FragCoord;

__attribute__((annotate("shady::builtin::Position")))
output native_vec4 gl_Position;

__attribute__((annotate("shady::builtin::WorkgroupId")))
__attribute__((address_space(389)))
native_uvec3 gl_WorkGroupID;

__attribute__((annotate("shady::builtin::VertexIndex")))
__attribute__((address_space(389)))
input int gl_VertexIndex;

__attribute__((annotate("shady::builtin::WorkgroupSize")))
__attribute__((address_space(389)))
native_uvec3 gl_WorkGroupSize;

__attribute__((annotate("shady::builtin::GlobalInvocationId")))
__attribute__((address_space(389)))
native_uvec3 gl_GlobalInvocationID;

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
}
#endif

#endif
