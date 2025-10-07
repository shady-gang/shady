#include "shady.h"
#include "shady_meta.h"

__shady_declare_builtin_type(spv_float, float)
__shady_declare_literal_i32(spv_0, 0)
__shady_declare_literal_i32(spv_1, 1)

__shady_declare_literal_string(spv_hi, "hi")

__shady_declare_param_ref(spv_param0, 0)
__shady_declare_param_ref(spv_param1, 1)

__shady_declare_ext_type(sampledImage2D, 25, /* sampled type */ __shady_ref(spv_float), /* dim = 2D */ __shady_ref(spv_1), /* depth */ __shady_ref(spv_0), /* arrayed */ __shady_ref(spv_0), /* multisample */ __shady_ref(spv_0), /* sampled = 1 */ __shady_ref(spv_1), /* image format = unknown */ __shady_ref(spv_0))
__shady_declare_ext_type(sampler2D, 27, __shady_ref(sampledImage2D))
sampler2D texSampler;

// declare the thing
typedef float native_vec4     __attribute__((ext_vector_type(4)));
typedef float native_vec3     __attribute__((ext_vector_type(3)));
typedef float native_vec2     __attribute__((ext_vector_type(2)));

__shady_declare_ext_inst(spv_texture2D, 87, __shady_ref(spv_param0), __shady_ref(spv_param1))
native_vec4 texture2D(const sampler2D, native_vec2) __shady_bind_ext_inst(spv_texture2D)

ray_generation_shader native_vec4 f(native_vec2 st) {
    return texture2D(texSampler, st);
}
