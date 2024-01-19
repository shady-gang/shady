#if defined(__cplusplus) & !defined(SHADY_CPP_NO_WRAPPER_CLASSES)
#define SHADY_ENABLE_WRAPPER_CLASSES
#endif

typedef float native_vec4     __attribute__((ext_vector_type(4)));
typedef float native_vec3     __attribute__((ext_vector_type(3)));
typedef float native_vec2     __attribute__((ext_vector_type(2)));

typedef int native_ivec4      __attribute__((ext_vector_type(4)));
typedef int native_ivec3      __attribute__((ext_vector_type(3)));
typedef int native_ivec2      __attribute__((ext_vector_type(2)));

typedef unsigned native_uvec4 __attribute__((ext_vector_type(4)));
typedef unsigned native_uvec3 __attribute__((ext_vector_type(3)));
typedef unsigned native_uvec2 __attribute__((ext_vector_type(2)));

#ifdef SHADY_ENABLE_WRAPPER_CLASSES
template<typename Native, typename T>
struct vec4_impl {
    using This = vec4_impl<Native, T>;
    float x, y, z, w;

    vec4_impl() {}
    vec4_impl(T scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}
    vec4_impl(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    vec4_impl(Native n) : x(n.x), y(n.y), z(n.z), w(n.w) {}

    operator Native() {
        return (Native) { x, y, z, w };
    }

    This operator +(This other) {
        return This(x + other.x, y + other.y, z + other.z, w + other.w);
    }
    This operator -(This other) {
        return This(x - other.x, y - other.y, z - other.z, w - other.w);
    }
    This operator *(This other) {
        return This(x * other.x, y * other.y, z * other.z, w * other.w);
    }
    This operator /(This other) {
        return This(x / other.x, y / other.y, z / other.z, w / other.w);
    }
    This operator *(T s) {
        return This(x * s, y * s, z * s, w * s);
    }
    This operator /(T s) {
        return This(x / s, y / s, z / s, z / s);
    }
};

template<typename Native, typename T>
struct vec3_impl {
    using This = vec3_impl<Native, T>;
    float x, y, z;

    vec3_impl() {}
    vec3_impl(T scalar) : x(scalar), y(scalar), z(scalar) {}
    vec3_impl(T x, T y, T z) : x(x), y(y), z(z) {}
    vec3_impl(Native n) : x(n.x), y(n.y), z(n.z) {}

    operator Native() {
        return (Native) { x, y, z };
    }

    This operator +(This other) {
        return This(x + other.x, y + other.y, z + other.z);
    }
    This operator -(This other) {
        return This(x - other.x, y - other.y, z - other.z);
    }
    This operator *(This other) {
        return This(x * other.x, y * other.y, z * other.z);
    }
    This operator /(This other) {
        return This(x / other.x, y / other.y, z / other.z);
    }
    This operator *(T s) {
        return This(x * s, y * s, z * s);
    }
    This operator /(T s) {
        return This(x / s, y / s, z / s);
    }
};

template<typename Native, typename T>
struct vec2_impl {
    using This = vec2_impl<Native, T>;
    float x, y;

    vec2_impl() {}
    vec2_impl(T scalar) : x(scalar), y(scalar) {}
    vec2_impl(T x, T y) : x(x), y(y) {}
    vec2_impl(Native n) : x(n.x), y(n.y) {}

    operator Native() {
        return (Native) { x, y };
    }

    This operator +(This other) {
        return This(x + other.x, y + other.y);
    }
    This operator -(This other) {
        return This(x - other.x, y - other.y);
    }
    This operator *(This other) {
        return This(x * other.x, y * other.y);
    }
    This operator /(This other) {
        return This(x / other.x, y / other.y);
    }
    This operator *(T s) {
        return This(x * s, y * s);
    }
    This operator /(T s) {
        return This(x / s, y / s);
    }
};

typedef vec4_impl<native_vec4, float> vec4;
typedef vec4_impl<native_uvec4, unsigned> uvec4;
typedef vec4_impl<native_ivec4, int> ivec4;

typedef vec3_impl<native_vec3, float> vec3;
typedef vec3_impl<native_uvec3, unsigned> uvec3;
typedef vec3_impl<native_ivec3, int> ivec3;

typedef vec2_impl<native_vec2, float> vec2;
typedef vec2_impl<native_uvec2, unsigned> uvec2;
typedef vec2_impl<native_ivec2, int> ivec2;
#else
typedef native_vec4 vec4;
typedef native_vec3 vec3;
typedef native_vec2 vec2;
typedef native_ivec4 ivec4;
typedef native_ivec3 ivec3;
typedef native_ivec2 ivec2;
typedef native_uvec4 uvec4;
typedef native_uvec3 uvec3;
typedef native_uvec2 uvec2;
#endif