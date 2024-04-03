#define FUNCTION __device__
#include "ao.c"

extern "C" {

__global__ void aobench_kernel(unsigned char* out) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    long int ptr = (long int) out;
    Ctx ctx = get_init_context();
    init_scene(&ctx);
    render_pixel(&ctx, x, y, WIDTH, HEIGHT, NSUBSAMPLES, out);
}

}