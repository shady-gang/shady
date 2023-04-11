#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable

#include "ao.c"

Scalar sqrtf(Scalar s) { return sqrt(s); }
Scalar floorf(Scalar s) { return floor(s); }
Scalar fabsf(Scalar s) { return fabs(s); }
Scalar sinf(Scalar s) { return sin(s); }
Scalar cosf(Scalar s) { return cos(s); }

global char zero;

__attribute__((reqd_work_group_size(16, 16, 1)))
kernel void aobench_kernel(global unsigned char* out) {
    Ctx ctx = get_init_context();
    init_scene(&ctx);

    auto x = get_global_id(0);
    auto y = get_global_id(1);

    render_pixel(&ctx, x, y, WIDTH, HEIGHT, NSUBSAMPLES, out);
    /*if (((x / 16) % 2) == ((y / 16) % 2)) {
        out[((y * HEIGHT) + x) * 3 + 0] = x;
        out[((y * HEIGHT) + x) * 3 + 1] = y;
        out[((y * HEIGHT) + x) * 3 + 2] = 0;
    } else {
        out[((y * HEIGHT) + x) * 3 + 0] = 255;
        out[((y * HEIGHT) + x) * 3 + 1] = zero;
        out[((y * HEIGHT) + x) * 3 + 2] = zero;
    }*/
    //out[index] = add(in1[index], in2[index]);
}