#include "ao.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void
saveppm(const char *fname, int w, int h, unsigned char *img)
{
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
}

void
render_host(unsigned char *img, int w, int h, int nsubsamples)
{
    int x, y;

    Float *fimg = (Float *)malloc(sizeof(Float) * w * h * 3);
    memset((void *)fimg, 0, sizeof(Float) * w * h * 3);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            render_pixel(x, y, w, h, nsubsamples, img, fimg);
        }
    }

}

int
main(int argc, char **argv)
{
    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    init_scene();
    render_host(img, WIDTH, HEIGHT, NSUBSAMPLES);
    saveppm("ao.ppm", WIDTH, HEIGHT, img);

    return 0;
}
