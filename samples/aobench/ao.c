//#include <math.h>

#include "ao.h"

unsigned int FNVHash(char* str, unsigned int length) {
    const unsigned int fnv_prime = 0x811C9DC5;
    unsigned int hash = 0;
    unsigned int i = 0;

    for (i = 0; i < length; str++, i++)
    {
        hash *= fnv_prime;
        hash ^= (*str);
    }

    return hash;
}

unsigned int nrand(unsigned int* rng) {
    unsigned int orand = *rng;
    *rng = FNVHash((char*) &orand, 4);
    return *rng;
}

Scalar drand48(Ctx* ctx) {
    Scalar n = (nrand(&ctx->rng) / 65536.0);
    n = n - floorf(n);
    return n;
}

static Scalar vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static void vcross(vec *c, vec v0, vec v1)
{

    c->x = v0.y * v1.z - v0.z * v1.y;
    c->y = v0.z * v1.x - v0.x * v1.z;
    c->z = v0.x * v1.y - v0.y * v1.x;
}

static void vnormalize(vec *c)
{
    Scalar length = sqrtf(vdot((*c), (*c)));

    if (fabsf(length) > 1.0e-17) {
        c->x /= length;
        c->y /= length;
        c->z /= length;
    }
}

void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs;

    rs.x = ray->org.x - sphere->center.x;
    rs.y = ray->org.y - sphere->center.y;
    rs.z = ray->org.z - sphere->center.z;

    Scalar B = vdot(rs, ray->dir);
    Scalar C = vdot(rs, rs) - sphere->radius * sphere->radius;
    Scalar D = B * B - C;

    if (D > 0.0) {
        Scalar t = -B - sqrtf(D);

        if ((t > 0.0) && (t < isect->t)) {
            isect->t = t;
            isect->hit = 1;

            isect->p.x = ray->org.x + ray->dir.x * t;
            isect->p.y = ray->org.y + ray->dir.y * t;
            isect->p.z = ray->org.z + ray->dir.z * t;

            isect->n.x = isect->p.x - sphere->center.x;
            isect->n.y = isect->p.y - sphere->center.y;
            isect->n.z = isect->p.z - sphere->center.z;

            vnormalize(&(isect->n));
        }
    }
}

void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    Scalar d = -vdot(plane->p, plane->n);
    Scalar v = vdot(ray->dir, plane->n);

    if (fabsf(v) < 1.0e-17) return;

    Scalar t = -(vdot(ray->org, plane->n) + d) / v;

    if ((t > 0.0) && (t < isect->t)) {
        isect->t = t;
        isect->hit = 1;

        isect->p.x = ray->org.x + ray->dir.x * t;
        isect->p.y = ray->org.y + ray->dir.y * t;
        isect->p.z = ray->org.z + ray->dir.z * t;

        isect->n = plane->n;
    }
}

void
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    vcross(&basis[0], basis[1], basis[2]);
    vnormalize(&basis[0]);

    vcross(&basis[1], basis[2], basis[0]);
    vnormalize(&basis[1]);
}

void ambient_occlusion(Ctx* ctx, vec *col, const Isect *isect)
{
    int    i, j;
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    Scalar eps = 0.0001;

    vec p;

    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;

    vec basis[3];
    orthoBasis(basis, isect->n);

    Scalar occlusion = 0.0;

    for (j = 0; j < ntheta; j++) {
        for (i = 0; i < nphi; i++) {
            Scalar theta = sqrtf(drand48(ctx));
            Scalar phi   = 2.0 * M_PI * drand48(ctx);

            Scalar x = cosf(phi) * theta;
            Scalar y = sinf(phi) * theta;
            Scalar z = sqrtf(1.0 - theta * theta);

            // local -> global
            Scalar rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            Scalar ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            Scalar rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

            Ray ray;

            ray.org = p;
            ray.dir.x = rx;
            ray.dir.y = ry;
            ray.dir.z = rz;

            Isect occIsect;
            occIsect.t   = 1.0e+17;
            occIsect.hit = 0;

            ray_sphere_intersect(&occIsect, &ray, &ctx->spheres[0]);
            ray_sphere_intersect(&occIsect, &ray, &ctx->spheres[1]);
            ray_sphere_intersect(&occIsect, &ray, &ctx->spheres[2]);
            ray_plane_intersect (&occIsect, &ray, &ctx->plane);

            if (occIsect.hit) occlusion += 1.0;

        }
    }

    occlusion = (ntheta * nphi - occlusion) / (Scalar)(ntheta * nphi);

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

unsigned char aobench_clamp(Scalar f)
{
    Scalar s = (f * 255.5);

    if (s < 0.f) return 0;
    if (s > 255.f) return 255;

    return (unsigned char) s;
}

void render_pixel(Ctx* ctx, int x, int y, int w, int h, int nsubsamples, unsigned char* img) {
    Scalar pixel[3] = { 0, 0, 0 };

    pixel[2] = 125.f;

    int u, v;
    for (v = 0; v < nsubsamples; v++) {
        for (u = 0; u < nsubsamples; u++) {
            Scalar px = (x + (u / (Scalar)nsubsamples) - (w / 2.0)) / (w / 2.0);
            Scalar py = -(y + (v / (Scalar)nsubsamples) - (h / 2.0)) / (h / 2.0);

            Ray ray = {};

            ray.org.x = 0.0;
            ray.org.y = 0.0;
            ray.org.z = 0.0;

            ray.dir.x = px;
            ray.dir.y = py;
            ray.dir.z = -1.0;
            vnormalize(&(ray.dir));

            Isect isect = {};
            isect.t   = 1.0e+17;
            isect.hit = 0;

            ray_sphere_intersect(&isect, &ray, &ctx->spheres[0]);
            ray_sphere_intersect(&isect, &ray, &ctx->spheres[1]);
            ray_sphere_intersect(&isect, &ray, &ctx->spheres[2]);
            ray_plane_intersect (&isect, &ray, &ctx->plane);

            pixel[0] = isect.t * 0.05f;
            pixel[1] = isect.t * 0.25f;
            pixel[2] = 0.f;

            // if (isect.hit) {
            //     vec col;
            //     ambient_occlusion(ctx, &col, &isect);
            //     pixel[0] += col.x;
            //     pixel[1] += col.y;
            //     pixel[2] += col.z;
            // }

        }
    }

    //pixel[1] = 255.f;

    pixel[0] /= (Scalar)(nsubsamples * nsubsamples);
    pixel[1] /= (Scalar)(nsubsamples * nsubsamples);
    pixel[2] /= (Scalar)(nsubsamples * nsubsamples);

    img[3 * (y * w + x) + 0] = aobench_clamp(pixel[0]);
    img[3 * (y * w + x) + 1] = aobench_clamp(pixel[1]);
    img[3 * (y * w + x) + 2] = aobench_clamp(pixel[2]);
}

void init_scene(Ctx* ctx)
{
    ctx->spheres[0].center.x = -2.0;
    ctx->spheres[0].center.y =  0.0;
    ctx->spheres[0].center.z = -3.5;
    ctx->spheres[0].radius = 0.5;

    ctx->spheres[1].center.x = -0.5;
    ctx->spheres[1].center.y =  0.0;
    ctx->spheres[1].center.z = -3.0;
    ctx->spheres[1].radius = 0.5;

    ctx->spheres[2].center.x =  1.0;
    ctx->spheres[2].center.y =  0.0;
    ctx->spheres[2].center.z = -2.2;
    ctx->spheres[2].radius = 0.5;

    ctx->plane.p.x = 0.0;
    ctx->plane.p.y = -0.5;
    ctx->plane.p.z = 0.0;

    ctx->plane.n.x = 0.0;
    ctx->plane.n.y = 1.0;
    ctx->plane.n.z = 0.0;

}

Ctx get_init_context() {
    return (Ctx) {
        .rng = 0xFEEFDEED,
    };
}