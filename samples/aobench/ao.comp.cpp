#include <stdint.h>

#define compute_shader __attribute__((annotate("shady::entry_point::Compute")))

#define location(i) __attribute__((annotate("shady::location::"#i)))

#define input __attribute__((address_space(389)))
#define output __attribute__((address_space(390)))
#define global __attribute__((address_space(1)))

typedef uint32_t uvec4 __attribute__((ext_vector_type(4)));
typedef float vec4 __attribute__((ext_vector_type(4)));

typedef uint32_t uvec3 __attribute__((ext_vector_type(3)));
typedef float vec3 __attribute__((ext_vector_type(3)));

/*__attribute__((annotate("shady::builtin::FragCoord")))
input vec4 fragCoord;

location(0) input vec3 fragColor;
location(0) output vec4 outColor;*/

__attribute__((annotate("shady::builtin::WorkgroupId")))
input uvec3 workgroup_id;

__attribute__((annotate("shady::builtin::GlobalInvocationId")))
input uvec3 global_id;

float sqrtf(float) __asm__("shady::prim_op::sqrt");
float sinf(float) __asm__("shady::prim_op::sin");
float cosf(float) __asm__("shady::prim_op::cos");
float fmodf(float, float) __asm__("shady::prim_op::mod");
float fabsf(float) __asm__("shady::prim_op::abs");
float floorf(float) __asm__("shady::prim_op::floor");
#include "ao.c"

extern "C" __attribute__((annotate("shady::workgroup_size::16::16::1")))
compute_shader void aobench_kernel(global unsigned char* out) {
    //outColor = (vec4) { fragColor[0], fragColor[1], fragColor[2], 1.0f };
    //outColor = (vec4) { fragCoord[0] / 1024, fragCoord[1] / 1024, 1.0f, 1.0f };

    Ctx ctx = get_init_context();
    init_scene(&ctx);

    int x = global_id.x;
    int y = global_id.y;
    //int x = (int) fragCoord.x % 1024;
    //int y = (int) fragCoord.y % 1024;

    // unsigned int out[3]; // = { 55, 0, 0};
    out[0] = 255;
    out[1] = 255;
    render_pixel(&ctx, x + 3, y, WIDTH, HEIGHT, NSUBSAMPLES, (unsigned char*) out);
    //out[2] = 155;
    // out[0] = x / 4;
    // out[1] = y / 4;
    //outColor = (vec4) { ((int) out[0]) / 255.0f, ((int) out[1]) / 255.0f, ((int) out[2]) / 255.0f, 1.0f };
}