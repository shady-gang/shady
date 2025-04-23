#ifndef GLSL_STD_H
#define GLSL_STD_H

#include "shady.h"

#ifdef __cplusplus
using namespace vcc;

namespace glsl_std {
#endif

static float round1f(float) __asm__("shady::pure_op::GLSL.std.450::1::Invocation");
static float roundEven1f(float) __asm__("shady::pure_op::GLSL.std.450::2::Invocation");
static float trunc1f(float) __asm__("shady::pure_op::GLSL.std.450::3::Invocation");
static float abs1f(float) __asm__("shady::pure_op::GLSL.std.450::4::Invocation");
static float abs1i(int) __asm__("shady::pure_op::GLSL.std.450::5::Invocation");
static float sign1f(float) __asm__("shady::pure_op::GLSL.std.450::6::Invocation");
static float sign1i(int) __asm__("shady::pure_op::GLSL.std.450::7::Invocation");
static float floor1f(float) __asm__("shady::pure_op::GLSL.std.450::8::Invocation");
static float ceil1f(float) __asm__("shady::pure_op::GLSL.std.450::9::Invocation");
static float fract1f(float) __asm__("shady::pure_op::GLSL.std.450::10::Invocation");
static float radians1f(float) __asm__("shady::pure_op::GLSL.std.450::11::Invocation");
static float degrees1f(float) __asm__("shady::pure_op::GLSL.std.450::12::Invocation");
static float sin1f(float) __asm__("shady::pure_op::GLSL.std.450::13::Invocation");
static float cos1f(float) __asm__("shady::pure_op::GLSL.std.450::14::Invocation");
static float tan1f(float) __asm__("shady::pure_op::GLSL.std.450::15::Invocation");
static float asin1f(float) __asm__("shady::pure_op::GLSL.std.450::16::Invocation");
static float acos1f(float) __asm__("shady::pure_op::GLSL.std.450::17::Invocation");
static float atan1f(float) __asm__("shady::pure_op::GLSL.std.450::18::Invocation");
static float sinh1f(float) __asm__("shady::pure_op::GLSL.std.450::19::Invocation");
static float cosh1f(float) __asm__("shady::pure_op::GLSL.std.450::20::Invocation");
static float tanh1f(float) __asm__("shady::pure_op::GLSL.std.450::21::Invocation");
static float asinh1f(float) __asm__("shady::pure_op::GLSL.std.450::22::Invocation");
static float acosh1f(float) __asm__("shady::pure_op::GLSL.std.450::23::Invocation");
static float atanh1f(float) __asm__("shady::pure_op::GLSL.std.450::24::Invocation");
static float atan2_1f(float) __asm__("shady::pure_op::GLSL.std.450::25::Invocation");
static float pow2f(float, float) __asm__("shady::pure_op::GLSL.std.450::26::Invocation");
static float exp1f(float) __asm__("shady::pure_op::GLSL.std.450::27::Invocation");
static float log1f(float) __asm__("shady::pure_op::GLSL.std.450::28::Invocation");
static float exp2_1f(float) __asm__("shady::pure_op::GLSL.std.450::29::Invocation");
static float log2_1f(float) __asm__("shady::pure_op::GLSL.std.450::30::Invocation");
static float sqrt1f(float) __asm__("shady::pure_op::GLSL.std.450::31::Invocation");
static float inverse_sqrt1f(float) __asm__("shady::pure_op::GLSL.std.450::32::Invocation");
//static float determinant_m4f(mat4) __asm__("shady::pure_op::GLSL.std.450::33::Invocation");
//static float matrix_inverse_m4f(mat4) __asm__("shady::pure_op::GLSL.std.450::34::Invocation");
static float modf_1f(float, float*) __asm__("shady::pure_op::GLSL.std.450::35::Invocation");
typedef struct { float f; float i; } modf_struct_1f_result_t;
static modf_struct_1f_result_t modf_struct_1f(float) __asm__("shady::pure_op::GLSL.std.450::36::Invocation");
static float min_2f(float, float) __asm__("shady::pure_op::GLSL.std.450::37::Invocation");
static float min_2u(unsigned, unsigned) __asm__("shady::pure_op::GLSL.std.450::38::Invocation");
static float min_2i(int, int) __asm__("shady::pure_op::GLSL.std.450::39::Invocation");
static float max_2f(float, float) __asm__("shady::pure_op::GLSL.std.450::40::Invocation");
static float max_2u(unsigned, unsigned) __asm__("shady::pure_op::GLSL.std.450::41::Invocation");
static float max_2i(int, int) __asm__("shady::pure_op::GLSL.std.450::42::Invocation");
static float clamp_1f(float, float, float) __asm__("shady::pure_op::GLSL.std.450::43::Invocation");
static float clamp_1u(unsigned, unsigned, unsigned) __asm__("shady::pure_op::GLSL.std.450::44::Invocation");
static float clamp_1i(int, int, int) __asm__("shady::pure_op::GLSL.std.450::45::Invocation");
static float mix_1f(float, float, float) __asm__("shady::pure_op::GLSL.std.450::46::Invocation");
static float step_1f(float, float) __asm__("shady::pure_op::GLSL.std.450::48::Invocation");
static float smoothstep_1f(float, float, float) __asm__("shady::pure_op::GLSL.std.450::49::Invocation");
static float fma_1f(float, float, float) __asm__("shady::pure_op::GLSL.std.450::50::Invocation");
static float frexp_1f(float, float*) __asm__("shady::pure_op::GLSL.std.450::51::Invocation");
typedef struct { float f; float exp; } frexp_struct_1f_result_t;
static frexp_struct_1f_result_t frexp_struct_1f(float) __asm__("shady::pure_op::GLSL.std.450::52::Invocation");
static float ldexp_1f(float, int) __asm__("shady::pure_op::GLSL.std.450::53::Invocation");

static unsigned pack_snorm_4x8(native_vec4) __asm__("shady::pure_op::GLSL.std.450::54::Invocation");
static unsigned pack_unorm_4x8(native_vec4) __asm__("shady::pure_op::GLSL.std.450::55::Invocation");
static unsigned pack_snorm_2x16(native_vec2) __asm__("shady::pure_op::GLSL.std.450::56::Invocation");
static unsigned pack_unorm_2x16(native_vec2) __asm__("shady::pure_op::GLSL.std.450::57::Invocation");

static native_vec2 unpack_snorm_2x16(unsigned) __asm__("shady::pure_op::GLSL.std.450::60::Invocation");
static native_vec2 unpack_unorm_2x16(unsigned) __asm__("shady::pure_op::GLSL.std.450::61::Invocation");
static native_vec4 unpack_snorm_4x8(unsigned) __asm__("shady::pure_op::GLSL.std.450::63::Invocation");
static native_vec4 unpack_unorm_4x8(unsigned) __asm__("shady::pure_op::GLSL.std.450::64::Invocation");

// TODO: half, double precision junk

static float length_vec2(native_vec2) __asm__("shady::pure_op::GLSL.std.450::66::Invocation");
static float length_vec3(native_vec3) __asm__("shady::pure_op::GLSL.std.450::66::Invocation");
static float length_vec4(native_vec4) __asm__("shady::pure_op::GLSL.std.450::66::Invocation");

static float distance_vec2(native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::67::Invocation");
static float distance_vec3(native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::67::Invocation");
static float distance_vec4(native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::67::Invocation");

static native_vec3 cross_vec3(native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::68::Invocation");

static native_vec2 normalize_vec2(native_vec2) __asm__("shady::pure_op::GLSL.std.450::69::Invocation");
static native_vec3 normalize_vec3(native_vec3) __asm__("shady::pure_op::GLSL.std.450::69::Invocation");
static native_vec4 normalize_vec4(native_vec4) __asm__("shady::pure_op::GLSL.std.450::69::Invocation");

static native_vec2 faceforward_vec2(native_vec2, native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::70::Invocation");
static native_vec3 faceforward_vec3(native_vec3, native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::70::Invocation");
static native_vec4 faceforward_vec4(native_vec4, native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::70::Invocation");

static native_vec2 reflect_vec2(native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::71::Invocation");
static native_vec3 reflect_vec3(native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::71::Invocation");
static native_vec4 reflect_vec4(native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::71::Invocation");

static native_vec2 refract_vec2(native_vec2, native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::72::Invocation");
static native_vec3 refract_vec3(native_vec3, native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::72::Invocation");
static native_vec4 refract_vec4(native_vec4, native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::72::Invocation");

static int findLSB_1u(unsigned) __asm__("shady::pure_op::GLSL.std.450::73::Invocation");

static int findMSB_1u(unsigned) __asm__("shady::pure_op::GLSL.std.450::74::Invocation");
static int findMSB_1i(int) __asm__("shady::pure_op::GLSL.std.450::75::Invocation");

static       float interpolateAtCentroid_1pf(      float*) __asm__("shady::pure_op::GLSL.std.450::76::Invocation");
static native_vec2 interpolateAtCentroid_2pf(native_vec2*) __asm__("shady::pure_op::GLSL.std.450::76::Invocation");
static native_vec3 interpolateAtCentroid_3pf(native_vec3*) __asm__("shady::pure_op::GLSL.std.450::76::Invocation");
static native_vec4 interpolateAtCentroid_4pf(native_vec4*) __asm__("shady::pure_op::GLSL.std.450::76::Invocation");

static       float interpolateAtSample_1pf(      float*, unsigned sample) __asm__("shady::pure_op::GLSL.std.450::77::Invocation");
static native_vec2 interpolateAtSample_2pf(native_vec2*, unsigned sample) __asm__("shady::pure_op::GLSL.std.450::77::Invocation");
static native_vec3 interpolateAtSample_3pf(native_vec3*, unsigned sample) __asm__("shady::pure_op::GLSL.std.450::77::Invocation");
static native_vec4 interpolateAtSample_4pf(native_vec4*, unsigned sample) __asm__("shady::pure_op::GLSL.std.450::77::Invocation");

static       float interpolateAtOffset_1pf(      float*, native_vec2 offset) __asm__("shady::pure_op::GLSL.std.450::78::Invocation");
static native_vec2 interpolateAtOffset_2pf(native_vec2*, native_vec2 offset) __asm__("shady::pure_op::GLSL.std.450::78::Invocation");
static native_vec3 interpolateAtOffset_3pf(native_vec3*, native_vec2 offset) __asm__("shady::pure_op::GLSL.std.450::78::Invocation");
static native_vec4 interpolateAtOffset_4pf(native_vec4*, native_vec2 offset) __asm__("shady::pure_op::GLSL.std.450::78::Invocation");

static       float nmin_1f(      float,       float) __asm__("shady::pure_op::GLSL.std.450::79::Invocation");
static native_vec2 nmin_2f(native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::79::Invocation");
static native_vec3 nmin_3f(native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::79::Invocation");
static native_vec4 nmin_4f(native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::79::Invocation");

static       float nmax_1f(      float,       float) __asm__("shady::pure_op::GLSL.std.450::80::Invocation");
static native_vec2 nmax_2f(native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::80::Invocation");
static native_vec3 nmax_3f(native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::80::Invocation");
static native_vec4 nmax_4f(native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::80::Invocation");

static       float nclamp_1f(      float,       float,       float) __asm__("shady::pure_op::GLSL.std.450::81::Invocation");
static native_vec2 nclamp_2f(native_vec2, native_vec2, native_vec2) __asm__("shady::pure_op::GLSL.std.450::81::Invocation");
static native_vec3 nclamp_3f(native_vec3, native_vec3, native_vec3) __asm__("shady::pure_op::GLSL.std.450::81::Invocation");
static native_vec4 nclamp_4f(native_vec4, native_vec4, native_vec4) __asm__("shady::pure_op::GLSL.std.450::81::Invocation");

#ifdef __cplusplus
}
#endif

#endif