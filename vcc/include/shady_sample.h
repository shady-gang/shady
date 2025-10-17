#ifndef SHADY_SAMPLE_H
#define SHADY_SAMPLE_H

#include "shady.h"
#include "shady_meta.h"

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
extern "C" {
namespace vcc {
#endif

__shady_declare_builtin_type(spv_float, float)
__shady_declare_literal_i32(spv_0, 0)
__shady_declare_literal_i32(spv_1, 1)
__shady_declare_literal_i32(spv_2, 2)
__shady_declare_literal_i32(spv_3, 3)

__shady_declare_param_ref(spv_param0, 0)
__shady_declare_param_ref(spv_param1, 1)

__shady_declare_ext_type(sampledImage1D,   25, /* sampled type */ __shady_ref(spv_float), /* dim =   1D */ __shady_ref(spv_0), /* depth */ __shady_ref(spv_0), /* arrayed */ __shady_ref(spv_0), /* multisample */ __shady_ref(spv_0), /* sampled = 1 */ __shady_ref(spv_1), /* image format = unknown */ __shady_ref(spv_0))
__shady_declare_ext_type(sampledImage2D,   25, /* sampled type */ __shady_ref(spv_float), /* dim =   2D */ __shady_ref(spv_1), /* depth */ __shady_ref(spv_0), /* arrayed */ __shady_ref(spv_0), /* multisample */ __shady_ref(spv_0), /* sampled = 1 */ __shady_ref(spv_1), /* image format = unknown */ __shady_ref(spv_0))
__shady_declare_ext_type(sampledImage3D,   25, /* sampled type */ __shady_ref(spv_float), /* dim =   3D */ __shady_ref(spv_2), /* depth */ __shady_ref(spv_0), /* arrayed */ __shady_ref(spv_0), /* multisample */ __shady_ref(spv_0), /* sampled = 1 */ __shady_ref(spv_1), /* image format = unknown */ __shady_ref(spv_0))
__shady_declare_ext_type(sampledImageCube, 25, /* sampled type */ __shady_ref(spv_float), /* dim = Cube */ __shady_ref(spv_3), /* depth */ __shady_ref(spv_0), /* arrayed */ __shady_ref(spv_0), /* multisample */ __shady_ref(spv_0), /* sampled = 1 */ __shady_ref(spv_1), /* image format = unknown */ __shady_ref(spv_0))

__shady_declare_ext_type(sampler1D,   27, __shady_ref(sampledImage1D))
__shady_declare_ext_type(sampler2D,   27, __shady_ref(sampledImage2D))
__shady_declare_ext_type(sampler3D,   27, __shady_ref(sampledImage3D))
__shady_declare_ext_type(samplerCube, 27, __shady_ref(sampledImageCube))

__shady_declare_ext_inst(spv_texture_sample, 87, __shady_ref(spv_param0), __shady_ref(spv_param1))

native_vec4   texture1D(const sampler1D,   float)       __shady_bind_ext_inst(spv_texture_sample)
native_vec4   texture2D(const sampler2D,   native_vec2) __shady_bind_ext_inst(spv_texture_sample)
native_vec4   texture3D(const sampler3D,   native_vec3) __shady_bind_ext_inst(spv_texture_sample)
native_vec4 textureCube(const samplerCube, native_vec3) __shady_bind_ext_inst(spv_texture_sample)

#if defined(__cplusplus)
static inline native_vec4 texture(const sampler1D img, float coords) { return texture1D(img, coords); }
static inline native_vec4 texture(const sampler2D img, native_vec2 coords) { return texture2D(img, coords); }
static inline native_vec4 texture(const sampler3D img, native_vec3 coords) { return texture3D(img, coords); }
static inline native_vec4 texture(const samplerCube img, native_vec3 coords) { return textureCube(img, coords); }
#endif

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
}
}
#endif

#endif