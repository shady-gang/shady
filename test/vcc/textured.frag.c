#include <stdint.h>
#include <shady.h>

descriptor_set(0) descriptor_binding(1) uniform_constant sampler2D texSampler;

location(0) input vec3 fragColor;
location(1) input vec2 fragTexCoord;

location(0) output vec4 outColor;

fragment_shader void main() {
    outColor = texture2D(texSampler, fragTexCoord) * (vec4) { fragColor.x * 2.5f, fragColor.y * 2.5f, fragColor.z * 2.5f, 1.0f };
}
