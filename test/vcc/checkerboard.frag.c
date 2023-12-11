#include <stdint.h>
#include <shady.h>

location(0) output vec4 fragColor;

fragment_shader void main() {
    int fillx = (((int) gl_FragCoord[0]) / 16) % 2;
    int filly = (((int) gl_FragCoord[1]) / 16) % 2;
    int fill = fillx ^ filly;
    fragColor = (vec4) {fill, fill, 0.0f, 1.0f };
}