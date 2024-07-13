#define __SHADY__

#include <memory>
#include "shady_vec.h"

using namespace vcc;

void check_native_casts(const vec4& v4, const uvec4& u4, const ivec4& i4) {
    native_vec4 nv4 = v4;
    native_uvec4 nu4 = u4;
    native_ivec4 ni4 = i4;
    vec4 rv4 = nv4;
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

void check_swizzle_const(const vec4& v) {
    float f = v.x;
    vec2 v2 = v.xy;
    //float err = v2.w;
}

void check_swizzle_mut(vec4& v) {
    v.x = 0.5f;
    v.xy = vec2(0.5f, 0.9f);
}

int main(int argc, char** argv) {
    {
        vec4 x;
    }
    std::unique_ptr<vec4> uptr;
}