#include <shady.h>

location(0) output vec4 fragColor;

vec4 red(void);

fragment_shader void main() {
    fragColor = red();
}