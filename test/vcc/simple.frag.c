#include <shady.h>

location(0) output vec4 fragColor;

fragment_shader void main() {
    fragColor = (vec4) { 0.0f, 0.5f, 1.0f, 1.0f };
}