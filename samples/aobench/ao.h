#define WIDTH        2048
#define HEIGHT       2048
#define NSUBSAMPLES  1
#define NAO_SAMPLES  8

typedef float Scalar;

#ifndef M_PI
#undef M_PI
#endif
// it must be a float literal
#define M_PI 3.14159f

#ifdef _MSC_VER
#include <math.h>
#else
Scalar sqrtf(Scalar);
Scalar floorf(Scalar);
Scalar fabsf(Scalar);
Scalar sinf(Scalar);
Scalar cosf(Scalar);
#endif

typedef struct _vec
{
    Scalar x;
    Scalar y;
    Scalar z;
} vec;


typedef struct _Isect
{
    Scalar t;
    vec    p;
    vec    n;
    int    hit;
} Isect;

typedef struct _Sphere
{
    vec    center;
    Scalar radius;

} Sphere;

typedef struct _Plane
{
    vec    p;
    vec    n;

} Plane;

typedef struct _Ray
{
    vec    org;
    vec    dir;
} Ray;

typedef struct {
    Sphere spheres[3];
    Plane plane;
    unsigned int rng;
} Ctx;

Ctx get_init_context();
void init_scene(Ctx*);
void render_pixel(Ctx*, int x, int y, int w, int h, int nsubsamples, unsigned char* img);
