#include <shady.h>

location(0) input native_vec3 vertexColor;
location(0) output native_vec4 outColor;

fragment_shader void test() {
    native_vec4 a;
    a.xyz = vertexColor;
    a.w = 1.0f;
    outColor = a;
}
