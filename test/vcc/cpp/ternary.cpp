#include <shady.h>
#include <stdint.h>

using namespace vcc;

extern "C" {

compute_shader local_size(1, 1, 1) void main(int8_t* done, native_vec3* out)
{
    native_vec3 v1{1.0f, 1.0f, 1.0f};
    native_vec3 v2{2.0f, 2.0f, 2.0f};

    out[0] = done[0] ? v1 : v2;
}

}