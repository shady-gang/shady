#define Mat4f mat4
#define Vec4f vec4
#define Vec3f vec3

typedef union {
    struct {
        // we use row-major ordering
        float m00, m01, m02, m03,
              m10, m11, m12, m13,
              m20, m21, m22, m23,
              m30, m31, m32, m33;
    };
    //Vec4f rows[4];
    float arr[16];
} Mat4f;

static const Mat4f identity_mat4f = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

Mat4f transpose_mat4f(Mat4f src) {
    return (Mat4f) {
        src.m00, src.m10, src.m20, src.m30,
        src.m01, src.m11, src.m21, src.m31,
        src.m02, src.m12, src.m22, src.m32,
        src.m03, src.m13, src.m23, src.m33,
    };
}

Mat4f invert_mat4(Mat4f m) {
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
    Mat4f r;
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

/*Mat4f perspective_mat4f(float a, float fov, float n, float f) {
    float pi = M_PI;
    float s = 1.0f / tanf(fov * 0.5f * (pi / 180.0f));
    return (Mat4f) {
        s / a, 0, 0, 0,
        0, s, 0, 0,
        0, 0, -f / (f - n), -1.f,
        0, 0, - (f * n) / (f - n), 0
    };
}*/

Mat4f translate_mat4f(Vec3f offset) {
    Mat4f m = identity_mat4f;
    m.m30 = offset.x;
    m.m31 = offset.y;
    m.m32 = offset.z;
    return m;
}

/*Mat4f rotate_axis_mat4f(unsigned int axis, float f) {
    Mat4f m = { 0 };
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

Mat4f mul_mat4f(Mat4f l, Mat4f r) {
    Mat4f dst = { 0 };
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

Vec4f mul_mat4f_vec4f(Mat4f l, Vec4f r) {
    float src[4] = { r.x, r.y, r.z, r.w };
    float dst[4];
#define a(i, j) m##i##j
#define t(bc, br, i) l.a(i, br) * src[i]
#define e(bc, br) dst[br] = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2) + t(bc, br, 3);
#define row(c) e(c, 0) e(c, 1) e(c, 2) e(c, 3)
#define genmul() row(0)
    genmul()
    return (Vec4f) { dst[0], dst[1], dst[2], dst[3] };
}

#if defined(__cplusplus)
#endif
