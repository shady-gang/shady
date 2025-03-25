#include <shady.h>

location(0) output native_vec4 fragColor;

native_vec4 red(void);

fragment_shader void main() {
    fragColor = red();
}