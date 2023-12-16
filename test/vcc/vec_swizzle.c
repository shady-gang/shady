#include <shady.h>

location(0) input vec3 vertexColor;
location(0) output vec4 outColor;

fragment_shader void test() {
    vec4 a;
    a.xyz = vertexColor;
    a.w = 1.0f;
    outColor = a;
}
