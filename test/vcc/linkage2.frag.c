#include <shady.h>

location(0) output vec4 fragColor;

typedef struct {
    float r, g, b;
} RGB;

RGB red(void);

fragment_shader void main() {
    RGB color = red();
    fragColor.x = color.r;
    fragColor.y = color.g;
    fragColor.z = color.b;
    fragColor.w = 1.0f;
}