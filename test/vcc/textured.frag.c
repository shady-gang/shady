#include <stdint.h>

#define fragment_shader __attribute__((annotate("shady::entry_point::Fragment")))

#define location(i)            __attribute__((annotate("shady::location::"#i)))
#define descriptor_set(i)      __attribute__((annotate("shady::descriptor_set::"#i)))
#define descriptor_binding(i)  __attribute__((annotate("shady::descriptor_binding::"#i)))
#define input                  __attribute__((address_space(389)))
#define output                 __attribute__((address_space(390)))
#define uniform                __attribute__((annotate("shady::uniform")))

typedef uint32_t uvec4 __attribute__((ext_vector_type(4)));
typedef float vec4 __attribute__((ext_vector_type(4)));

typedef uint32_t uvec3 __attribute__((ext_vector_type(3)));
typedef float vec3 __attribute__((ext_vector_type(3)));

typedef uint32_t uvec2 __attribute__((ext_vector_type(2)));
typedef float vec2 __attribute__((ext_vector_type(2)));

typedef struct __shady_builtin_sampler2D {} sampler2D;

vec4 texture2D(const sampler2D, vec2) __asm__("shady::prim_op::sample_texture");

__attribute__((annotate("shady::builtin::FragCoord")))
input vec4 gl_FragCoord;

descriptor_set(0) descriptor_binding(1) uniform sampler2D texSampler;

location(0) input vec3 fragColor;
location(1) input vec2 fragTexCoord;

location(0) output vec4 outColor;

fragment_shader void main() {
    outColor = texture2D(texSampler, fragTexCoord) * (vec4) { fragColor.x * 2.5f, fragColor.y * 2.5f, fragColor.z * 2.5f, 1.0f };
}
