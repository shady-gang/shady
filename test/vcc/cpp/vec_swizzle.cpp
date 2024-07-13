#include <shady.h>

using namespace vcc;

extern "C" {

location(0) vec3 vertexColor;
location(0) vec4 outColor;

fragment_shader void test() {
    vec4 a;
    a.xyz = vertexColor;
    a.w = 1.0f;
    outColor = a;
}

}