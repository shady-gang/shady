#include <shady.h>

input vec3 vertexColor;
output vec4 outColor;

fragment_shader void test() {
    vec4 a;
    a.xyz = vertexColor;
    a.w = 1.0f;
    outColor = a;
}
