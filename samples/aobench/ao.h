#define WIDTH        1024
#define HEIGHT       1024
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8

typedef float Float;

#define M_PI 3.14159

Float sqrtf(Float);
Float floorf(Float);
Float fabsf(Float);
Float sinf(Float);
Float cosf(Float);

void init_scene();
void render_pixel(int x, int y, int w, int h, int nsubsamples, unsigned char* img, Float* fimg);
