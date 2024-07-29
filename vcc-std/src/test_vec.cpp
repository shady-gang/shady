#define __SHADY__

#include <memory>
#include "shady_vec.h"

using namespace vcc;

void check_native_casts(const vec4& v4, const uvec4& u4, const ivec4& i4) {
    native_vec4 nv4 = v4;
    native_uvec4 nu4 = u4;
    native_ivec4 ni4 = i4;
    vec4 rv4 = nv4;
    native_vec3 nv3;
    nv3 = v4.xyz;
}

void check_vector_scalar_ctors() {
    vec4 x4 = vec4(0.5f);
    vec4 y4 = { 0.5f };
    vec4 z4(0.5f);
    vec4 w4 = 0.5f;

    vec3 x3 = vec3(0.5f);
    vec3 y3 = { 0.5f };
    vec3 z3(0.5f);
    vec3 w3 = 0.5f;

    vec2 x2 = vec2(0.5f);
    vec2 y2 = { 0.5f };
    vec2 z2(0.5f);
    vec2 w2 = 0.5f;
}

void check_swizzle_const(const vec4& v4, const uvec4& u4, const ivec4& i4) {
    v4.x;
    v4.xy;
    v4.xyz;
    v4.xyzw;

    v4.xxxx;
    v4.xyww;
}

void check_ctor_weird() {
    vec4(vec2(0.5f), vec2(0.5f));
    vec4(0.5f, vec2(0.5f), 0.5f);
    vec4(0.5f, vec3(0.5f));
    vec4(vec3(0.5f), 0.5f);
}

void check_swizzle_mut(vec4& v) {
    v.x = 0.5f;
    v.xy = vec2(0.5f, 0.9f);
}

#include <cassert>
#include <cstdio>
int main(int argc, char** argv) {
    vec4 v(1.0f, 0.5f, 0.0f, -1.0f);
    float f;
    f = v.x; printf("f = %f;\n", f); assert(f == 1.0f);
    f = v.y; printf("f = %f;\n", f); assert(f == 0.5f);
    f = v.z; printf("f = %f;\n", f); assert(f == 0.0f);
    f = v.w; printf("f = %f;\n", f); assert(f == -1.0f);
    std::unique_ptr<vec4> uptr;
}