#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
namespace vcc {
#endif

typedef union mat4_ mat4;

static inline mat4 transpose_mat4(mat4 src);
static inline mat4 mul_mat4(mat4 l, mat4 r);
static inline vec4 mul_mat4_vec4f(mat4 l, vec4 r);

union mat4_ {
    struct {
        // we use row-major ordering
        float m00, m01, m02, m03,
              m10, m11, m12, m13,
              m20, m21, m22, m23,
              m30, m31, m32, m33;
    };
    //vec4 rows[4];
    float arr[16];


#if defined(__cplusplus)
    mat4 operator*(const mat4& other) {
        return mul_mat4(*this, other);
    }

    vec4 operator*(const vec4& other) {
        return mul_mat4_vec4f(*this, other);
    }
#endif
};

static const mat4 identity_mat4 = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

static inline mat4 transpose_mat4(mat4 src) {
    return (mat4) {
        src.m00, src.m10, src.m20, src.m30,
        src.m01, src.m11, src.m21, src.m31,
        src.m02, src.m12, src.m22, src.m32,
        src.m03, src.m13, src.m23, src.m33,
    };
}

static inline mat4 invert_mat4(mat4 m) {
    float a = m.m00 * m.m11 - m.m01 * m.m10;
    float b = m.m00 * m.m12 - m.m02 * m.m10;
    float c = m.m00 * m.m13 - m.m03 * m.m10;
    float d = m.m01 * m.m12 - m.m02 * m.m11;
    float e = m.m01 * m.m13 - m.m03 * m.m11;
    float f = m.m02 * m.m13 - m.m03 * m.m12;
    float g = m.m20 * m.m31 - m.m21 * m.m30;
    float h = m.m20 * m.m32 - m.m22 * m.m30;
    float i = m.m20 * m.m33 - m.m23 * m.m30;
    float j = m.m21 * m.m32 - m.m22 * m.m31;
    float k = m.m21 * m.m33 - m.m23 * m.m31;
    float l = m.m22 * m.m33 - m.m23 * m.m32;
    float det = a * l - b * k + c * j + d * i - e * h + f * g;
    det = 1.0f / det;
    mat4 r;
    r.m00 = ( m.m11 * l - m.m12 * k + m.m13 * j) * det;
    r.m01 = (-m.m01 * l + m.m02 * k - m.m03 * j) * det;
    r.m02 = ( m.m31 * f - m.m32 * e + m.m33 * d) * det;
    r.m03 = (-m.m21 * f + m.m22 * e - m.m23 * d) * det;
    r.m10 = (-m.m10 * l + m.m12 * i - m.m13 * h) * det;
    r.m11 = ( m.m00 * l - m.m02 * i + m.m03 * h) * det;
    r.m12 = (-m.m30 * f + m.m32 * c - m.m33 * b) * det;
    r.m13 = ( m.m20 * f - m.m22 * c + m.m23 * b) * det;
    r.m20 = ( m.m10 * k - m.m11 * i + m.m13 * g) * det;
    r.m21 = (-m.m00 * k + m.m01 * i - m.m03 * g) * det;
    r.m22 = ( m.m30 * e - m.m31 * c + m.m33 * a) * det;
    r.m23 = (-m.m20 * e + m.m21 * c - m.m23 * a) * det;
    r.m30 = (-m.m10 * j + m.m11 * h - m.m12 * g) * det;
    r.m31 = ( m.m00 * j - m.m01 * h + m.m02 * g) * det;
    r.m32 = (-m.m30 * d + m.m31 * b - m.m32 * a) * det;
    r.m33 = ( m.m20 * d - m.m21 * b + m.m22 * a) * det;
    return r;
}

/*mat4 perspective_mat4(float a, float fov, float n, float f) {
    float pi = M_PI;
    float s = 1.0f / tanf(fov * 0.5f * (pi / 180.0f));
    return (mat4) {
        s / a, 0, 0, 0,
        0, s, 0, 0,
        0, 0, -f / (f - n), -1.f,
        0, 0, - (f * n) / (f - n), 0
    };
}*/

static inline mat4 translate_mat4(vec3 offset) {
    mat4 m = identity_mat4;
    m.m30 = offset.x;
    m.m31 = offset.y;
    m.m32 = offset.z;
    return m;
}

/*mat4 rotate_axis_mat4(unsigned int axis, float f) {
    mat4 m = { 0 };
    m.m33 = 1;

    unsigned int t = (axis + 2) % 3;
    unsigned int s = (axis + 1) % 3;

    m.rows[t].arr[t] =  cosf(f);
    m.rows[t].arr[s] = -sinf(f);
    m.rows[s].arr[t] =  sinf(f);
    m.rows[s].arr[s] =  cosf(f);

    // leave that unchanged
    m.rows[axis].arr[axis] = 1;

    return m;
}*/

static inline mat4 mul_mat4(mat4 l, mat4 r) {
    mat4 dst = { 0 };
#define a(i, j) m##i##j
#define t(bc, br, i) l.a(i, br) * r.a(bc, i)
#define e(bc, br) dst.a(bc, br) = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2) + t(bc, br, 3);
#define row(c) e(c, 0) e(c, 1) e(c, 2) e(c, 3)
#define genmul() row(0) row(1) row(2) row(3)
    genmul()
    return dst;
#undef a
#undef t
#undef e
#undef row
#undef genmul
}

static inline vec4 mul_mat4_vec4f(mat4 l, vec4 r) {
    float src[4] = { r.x, r.y, r.z, r.w };
    float dst[4];
#define a(i, j) m##i##j
#define t(bc, br, i) l.a(i, br) * src[i]
#define e(bc, br) dst[br] = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2) + t(bc, br, 3);
#define row(c) e(c, 0) e(c, 1) e(c, 2) e(c, 3)
#define genmul() row(0)
    genmul()
    return (vec4) { dst[0], dst[1], dst[2], dst[3] };
}

typedef union {
    struct {
        // we use row-major ordering
        float m00, m01, m02,
              m10, m11, m12,
              m20, m21, m22;
    };
    //vec4 rows[4];
    float arr[9];
} Mat3f;

static const Mat3f identity_mat3f = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
};

static Mat3f transpose_mat3f(Mat3f src) {
    return (Mat3f) {
        src.m00, src.m10, src.m20,
        src.m01, src.m11, src.m21,
        src.m02, src.m12, src.m22,
    };
}

static Mat3f mul_mat3f(Mat3f l, Mat3f r) {
    Mat3f dst = { 0 };
#define a(i, j) m##i##j
#define t(bc, br, i) l.a(i, br) * r.a(bc, i)
#define e(bc, br) dst.a(bc, br) = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2);
#define row(c) e(c, 0) e(c, 1) e(c, 2)
#define genmul() row(0) row(1) row(2)
    genmul()
    return dst;
#undef a
#undef t
#undef e
#undef row
#undef genmul
}

typedef Mat3f mat3;

#if defined(__cplusplus) & !defined(SHADY_CPP_NO_NAMESPACE)
}
#endif
