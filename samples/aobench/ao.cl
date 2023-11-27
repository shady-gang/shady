#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable

#include "ao.c"

Scalar sqrtf(Scalar s) { return sqrt(s); }
Scalar floorf(Scalar s) { return floor(s); }
Scalar fabsf(Scalar s) { return fabs(s); }
Scalar sinf(Scalar s) { return sin(s); }
Scalar cosf(Scalar s) { return cos(s); }

global char zero;

extern "C" {

void debug_printf_i64(const __constant char*, long int) __asm__("__shady::prim_op::debug_printf::i64");
void debug_printf_i32_i32(const __constant char*, int, int) __asm__("__shady::prim_op::debug_printf::i32_i32");

}

__attribute__((reqd_work_group_size(16, 16, 1)))
kernel void aobench_kernel(global unsigned char* out) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    long int ptr = (long int) out;
    //debug_printf_i64("ptr: %lu\n", ptr);
    //debug_printf_i32_i32("ptr: %d %d\n", (int) (ptr << 32), (int) ptr);

    Ctx ctx = get_init_context();
    init_scene(&ctx);

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