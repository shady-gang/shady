#define EXTERNAL_FN static inline __device__ __attribute__((always_inline))
#define FUNCTION static inline __device__ __attribute__((always_inline))

#include "ao.c"

extern "C" {

__global__ void aobench_kernel(unsigned TEXEL_T* out) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    Ctx ctx = get_init_context();
    init_scene(&ctx);
    render_pixel(&ctx, x, y, WIDTH, HEIGHT, NSUBSAMPLES, out);
    // out[3 * (y * 2048 + x) + 0] = 255;
    // out[3 * (y * 2048 + x) + 1] = 255;
    // out[3 * (y * 2048 + x) + 2] = 255;
}

}