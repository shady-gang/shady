#ifndef _SHADY_H
#define _SHADY_H

#ifndef __SHADY__
#error "This header can only be used with Vcc"
#endif

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
extern "C" {
namespace vcc {
#endif

#include <stdint.h>

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

// implicitly sized vectors

typedef float native_vec4           __attribute__((ext_vector_type(4)));
typedef float native_vec3           __attribute__((ext_vector_type(3)));
typedef float native_vec2           __attribute__((ext_vector_type(2)));

typedef double native_dvec4         __attribute__((ext_vector_type(4)));
typedef double native_dvec3         __attribute__((ext_vector_type(3)));
typedef double native_dvec2         __attribute__((ext_vector_type(2)));

typedef int native_ivec4            __attribute__((ext_vector_type(4)));
typedef int native_ivec3            __attribute__((ext_vector_type(3)));
typedef int native_ivec2            __attribute__((ext_vector_type(2)));

typedef unsigned native_uvec4       __attribute__((ext_vector_type(4)));
typedef unsigned native_uvec3       __attribute__((ext_vector_type(3)));
typedef unsigned native_uvec2       __attribute__((ext_vector_type(2)));

// implicitly sized matrices

typedef float native_mat4           __attribute__((ext_vector_type(16))); typedef native_mat4 native_mat4x4;
typedef float native_mat3           __attribute__((ext_vector_type(9)));  typedef native_mat3 native_mat3x3;
typedef float native_mat2           __attribute__((ext_vector_type(4)));  typedef native_mat2 native_mat2x2;
typedef float native_mat4x3         __attribute__((ext_vector_type(12)));
typedef float native_mat4x2         __attribute__((ext_vector_type(8)));
typedef float native_mat3           __attribute__((ext_vector_type(9)));
typedef float native_mat3x2         __attribute__((ext_vector_type(6)));
typedef float native_mat2           __attribute__((ext_vector_type(4)));

typedef double native_dmat4         __attribute__((ext_vector_type(16))); typedef native_dmat4 native_dmat4x4;
typedef double native_dmat3         __attribute__((ext_vector_type(9)));  typedef native_dmat3 native_dmat3x3;
typedef double native_dmat2         __attribute__((ext_vector_type(4)));  typedef native_dmat2 native_dmat2x2;
typedef double native_dmat4x3       __attribute__((ext_vector_type(12)));
typedef double native_dmat4x2       __attribute__((ext_vector_type(8)));
typedef double native_dmat3         __attribute__((ext_vector_type(9)));
typedef double native_dmat3x2       __attribute__((ext_vector_type(6)));
typedef double native_dmat2         __attribute__((ext_vector_type(4)));

// explicitly sized vectors

typedef _Float16 native_f16vec4     __attribute__((ext_vector_type(4)));
typedef _Float16 native_f16vec3     __attribute__((ext_vector_type(3)));
typedef _Float16 native_f16vec2     __attribute__((ext_vector_type(2)));

typedef float native_f32vec4        __attribute__((ext_vector_type(4)));
typedef float native_f32vec3        __attribute__((ext_vector_type(3)));
typedef float native_f32vec2        __attribute__((ext_vector_type(2)));

typedef double native_f64vec4       __attribute__((ext_vector_type(4)));
typedef double native_f64vec3       __attribute__((ext_vector_type(3)));
typedef double native_f64vec2       __attribute__((ext_vector_type(2)));

typedef int8_t native_u8vec4        __attribute__((ext_vector_type(4)));
typedef int8_t native_u8vec3        __attribute__((ext_vector_type(3)));
typedef int8_t native_u8vec2        __attribute__((ext_vector_type(2)));

typedef int16_t native_u16vec4      __attribute__((ext_vector_type(4)));
typedef int16_t native_u16vec3      __attribute__((ext_vector_type(3)));
typedef int16_t native_u16vec2      __attribute__((ext_vector_type(2)));

typedef int32_t native_u32vec4      __attribute__((ext_vector_type(4)));
typedef int32_t native_u32vec3      __attribute__((ext_vector_type(3)));
typedef int32_t native_u32vec2      __attribute__((ext_vector_type(2)));

typedef int64_t native_u64vec4      __attribute__((ext_vector_type(4)));
typedef int64_t native_u64vec3      __attribute__((ext_vector_type(3)));
typedef int64_t native_u64vec2      __attribute__((ext_vector_type(2)));

typedef int8_t native_i8vec4        __attribute__((ext_vector_type(4)));
typedef int8_t native_i8vec3        __attribute__((ext_vector_type(3)));
typedef int8_t native_i8vec2        __attribute__((ext_vector_type(2)));

typedef int16_t native_i16vec4      __attribute__((ext_vector_type(4)));
typedef int16_t native_i16vec3      __attribute__((ext_vector_type(3)));
typedef int16_t native_i16vec2      __attribute__((ext_vector_type(2)));

typedef int32_t native_i32vec4      __attribute__((ext_vector_type(4)));
typedef int32_t native_i32vec3      __attribute__((ext_vector_type(3)));
typedef int32_t native_i32vec2      __attribute__((ext_vector_type(2)));

typedef int64_t native_i64vec4      __attribute__((ext_vector_type(4)));
typedef int64_t native_i64vec3      __attribute__((ext_vector_type(3)));
typedef int64_t native_i64vec2      __attribute__((ext_vector_type(2)));

// explicitly sized matrices

typedef _Float16 native_f16mat4     __attribute__((ext_vector_type(16))); typedef native_f16mat4 native_f16mat4x4;
typedef _Float16 native_f16mat3     __attribute__((ext_vector_type(9)));  typedef native_f16mat3 native_f16mat3x3;
typedef _Float16 native_f16mat2     __attribute__((ext_vector_type(4)));  typedef native_f16mat2 native_f16mat2x2;
typedef _Float16 native_f16mat4x3   __attribute__((ext_vector_type(12)));
typedef _Float16 native_f16mat4x2   __attribute__((ext_vector_type(8)));
typedef _Float16 native_f16mat3     __attribute__((ext_vector_type(9)));
typedef _Float16 native_f16mat3x2   __attribute__((ext_vector_type(6)));
typedef _Float16 native_f16mat2     __attribute__((ext_vector_type(4)));

typedef float native_f32mat4        __attribute__((ext_vector_type(16))); typedef native_f32mat4 native_f32mat4x4;
typedef float native_f32mat3        __attribute__((ext_vector_type(9)));  typedef native_f32mat3 native_f32mat3x3;
typedef float native_f32mat2        __attribute__((ext_vector_type(4)));  typedef native_f32mat2 native_f32mat2x2;
typedef float native_f32mat4x3      __attribute__((ext_vector_type(12)));
typedef float native_f32mat4x2      __attribute__((ext_vector_type(8)));
typedef float native_f32mat3        __attribute__((ext_vector_type(9)));
typedef float native_f32mat3x2      __attribute__((ext_vector_type(6)));
typedef float native_f32mat2        __attribute__((ext_vector_type(4)));

typedef double native_f64mat4       __attribute__((ext_vector_type(16))); typedef native_f64mat4 native_f64mat4x4;
typedef double native_f64mat3       __attribute__((ext_vector_type(9)));  typedef native_f64mat3 native_f64mat3x3;
typedef double native_f64mat2       __attribute__((ext_vector_type(4)));  typedef native_f64mat2 native_f64mat2x2;
typedef double native_f64mat4x3     __attribute__((ext_vector_type(12)));
typedef double native_f64mat4x2     __attribute__((ext_vector_type(8)));
typedef double native_f64mat3       __attribute__((ext_vector_type(9)));
typedef double native_f64mat3x2     __attribute__((ext_vector_type(6)));
typedef double native_f64mat2       __attribute__((ext_vector_type(4)));

typedef uint8_t native_u8mat4       __attribute__((ext_vector_type(16))); typedef native_u8mat4 native_u8mat4x4;
typedef uint8_t native_u8mat3       __attribute__((ext_vector_type(9)));  typedef native_u8mat3 native_u8mat3x3;
typedef uint8_t native_u8mat2       __attribute__((ext_vector_type(4)));  typedef native_u8mat2 native_u8mat2x2;
typedef uint8_t native_u8mat4x3     __attribute__((ext_vector_type(12)));
typedef uint8_t native_u8mat4x2     __attribute__((ext_vector_type(8)));
typedef uint8_t native_u8mat3       __attribute__((ext_vector_type(9)));
typedef uint8_t native_u8mat3x2     __attribute__((ext_vector_type(6)));
typedef uint8_t native_u8mat2       __attribute__((ext_vector_type(4)));

typedef uint16_t native_u16mat4     __attribute__((ext_vector_type(16))); typedef native_u16mat4 native_u16mat4x4;
typedef uint16_t native_u16mat3     __attribute__((ext_vector_type(9)));  typedef native_u16mat3 native_u16mat3x3;
typedef uint16_t native_u16mat2     __attribute__((ext_vector_type(4)));  typedef native_u16mat2 native_u16mat2x2;
typedef uint16_t native_u16mat4x3   __attribute__((ext_vector_type(12)));
typedef uint16_t native_u16mat4x2   __attribute__((ext_vector_type(8)));
typedef uint16_t native_u16mat3     __attribute__((ext_vector_type(9)));
typedef uint16_t native_u16mat3x2   __attribute__((ext_vector_type(6)));
typedef uint16_t native_u16mat2     __attribute__((ext_vector_type(4)));

typedef uint32_t native_u32mat4     __attribute__((ext_vector_type(16))); typedef native_u32mat4 native_u32mat4x4;
typedef uint32_t native_u32mat3     __attribute__((ext_vector_type(9)));  typedef native_u32mat3 native_u32mat3x3;
typedef uint32_t native_u32mat2     __attribute__((ext_vector_type(4)));  typedef native_u32mat2 native_u32mat2x2;
typedef uint32_t native_u32mat4x3   __attribute__((ext_vector_type(12)));
typedef uint32_t native_u32mat4x2   __attribute__((ext_vector_type(8)));
typedef uint32_t native_u32mat3     __attribute__((ext_vector_type(9)));
typedef uint32_t native_u32mat3x2   __attribute__((ext_vector_type(6)));
typedef uint32_t native_u32mat2     __attribute__((ext_vector_type(4)));

typedef uint64_t native_u64mat4     __attribute__((ext_vector_type(16))); typedef native_u64mat4 native_u64mat4x4;
typedef uint64_t native_u64mat3     __attribute__((ext_vector_type(9)));  typedef native_u64mat3 native_u64mat3x3;
typedef uint64_t native_u64mat2     __attribute__((ext_vector_type(4)));  typedef native_u64mat2 native_u64mat2x2;
typedef uint64_t native_u64mat4x3   __attribute__((ext_vector_type(12)));
typedef uint64_t native_u64mat4x2   __attribute__((ext_vector_type(8)));
typedef uint64_t native_u64mat3     __attribute__((ext_vector_type(9)));
typedef uint64_t native_u64mat3x2   __attribute__((ext_vector_type(6)));
typedef uint64_t native_u64mat2     __attribute__((ext_vector_type(4)));

typedef uint8_t native_i8mat4       __attribute__((ext_vector_type(16))); typedef native_i8mat4 native_i8mat4x4;
typedef uint8_t native_i8mat3       __attribute__((ext_vector_type(9)));  typedef native_i8mat3 native_i8mat3x3;
typedef uint8_t native_i8mat2       __attribute__((ext_vector_type(4)));  typedef native_i8mat2 native_i8mat2x2;
typedef uint8_t native_i8mat4x3     __attribute__((ext_vector_type(12)));
typedef uint8_t native_i8mat4x2     __attribute__((ext_vector_type(8)));
typedef uint8_t native_i8mat3       __attribute__((ext_vector_type(9)));
typedef uint8_t native_i8mat3x2     __attribute__((ext_vector_type(6)));
typedef uint8_t native_i8mat2       __attribute__((ext_vector_type(4)));

typedef uint16_t native_i16mat4     __attribute__((ext_vector_type(16))); typedef native_i16mat4 native_i16mat4x4;
typedef uint16_t native_i16mat3     __attribute__((ext_vector_type(9)));  typedef native_i16mat3 native_i16mat3x3;
typedef uint16_t native_i16mat2     __attribute__((ext_vector_type(4)));  typedef native_i16mat2 native_i16mat2x2;
typedef uint16_t native_i16mat4x3   __attribute__((ext_vector_type(12)));
typedef uint16_t native_i16mat4x2   __attribute__((ext_vector_type(8)));
typedef uint16_t native_i16mat3     __attribute__((ext_vector_type(9)));
typedef uint16_t native_i16mat3x2   __attribute__((ext_vector_type(6)));
typedef uint16_t native_i16mat2     __attribute__((ext_vector_type(4)));

typedef uint32_t native_i32mat4     __attribute__((ext_vector_type(16))); typedef native_i32mat4 native_i32mat4x4;
typedef uint32_t native_i32mat3     __attribute__((ext_vector_type(9)));  typedef native_i32mat3 native_i32mat3x3;
typedef uint32_t native_i32mat2     __attribute__((ext_vector_type(4)));  typedef native_i32mat2 native_i32mat2x2;
typedef uint32_t native_i32mat4x3   __attribute__((ext_vector_type(12)));
typedef uint32_t native_i32mat4x2   __attribute__((ext_vector_type(8)));
typedef uint32_t native_i32mat3     __attribute__((ext_vector_type(9)));
typedef uint32_t native_i32mat3x2   __attribute__((ext_vector_type(6)));
typedef uint32_t native_i32mat2     __attribute__((ext_vector_type(4)));

typedef uint64_t native_i64mat4     __attribute__((ext_vector_type(16))); typedef native_i64mat4 native_i64mat4x4;
typedef uint64_t native_i64mat3     __attribute__((ext_vector_type(9)));  typedef native_i64mat3 native_i64mat3x3;
typedef uint64_t native_i64mat2     __attribute__((ext_vector_type(4)));  typedef native_i64mat2 native_i64mat2x2;
typedef uint64_t native_i64mat4x3   __attribute__((ext_vector_type(12)));
typedef uint64_t native_i64mat4x2   __attribute__((ext_vector_type(8)));
typedef uint64_t native_i64mat3     __attribute__((ext_vector_type(9)));
typedef uint64_t native_i64mat3x2   __attribute__((ext_vector_type(6)));
typedef uint64_t native_i64mat2     __attribute__((ext_vector_type(4)));

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
}
#endif

#include "shady_sample.h"

#endif
