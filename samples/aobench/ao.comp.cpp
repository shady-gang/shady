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

#define EXTERNAL_FN static
#define FUNCTION static

#include "ao.c"

#define xstr(s) str(s)
#define str(s) #s

extern "C" __attribute__((annotate("shady::workgroup_size::" xstr(BLOCK_SIZE) "::" xstr(BLOCK_SIZE) "::1")))
compute_shader void aobench_kernel(global TEXEL_T* out) {
    Ctx ctx = get_init_context();
    init_scene(&ctx);

    int x = global_id.x;
    int y = global_id.y;
    render_pixel(&ctx, x, y, WIDTH, HEIGHT, NSUBSAMPLES, (TEXEL_T*) out);
}
