#include "ao.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "shady/runtime.h"
#include "shady/cli.h"

#include "log.h"
#include "list.h"
#include "util.h"

void saveppm(const char *fname, int w, int h, unsigned char *img) {
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
}

void render_host(unsigned char *img, int w, int h, int nsubsamples) {
    int x, y;
    Scalar* fimg = (Scalar *)malloc(sizeof(Scalar) * w * h * 3);
    memset((void *)fimg, 0, sizeof(Scalar) * w * h * 3);

    Ctx ctx = get_init_context();
    init_scene(&ctx);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            render_pixel(&ctx, x, y, w, h, nsubsamples, img);
        }
    }
}

#ifdef ENABLE_ISPC
extern void (aobench_kernel)(uint8_t*);

typedef struct {
    int32_t x, y, z;
} Vec3i;

int32_t subgroup_id;
Vec3i global_id;
Vec3i workgroup_size;
Vec3i workgroup_num;

void render_ispc(unsigned char *img, int w, int h, int nsubsamples) {
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    for (size_t y = 0; y < h / 16; y++) {
        for (size_t x = 0; x < w / 16; x++) {
            aobench_kernel(img);
        }
    }
}
#endif

void render_device(unsigned char *img, int w, int h, int nsubsamples, String path) {
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    set_log_level(INFO);
    CompilerConfig compiler_config = default_compiler_config();

    RuntimeConfig runtime_config = (RuntimeConfig) {
            .use_validation = true,
            .dump_spv = true,
    };

    info_print("Shady checkerboard test starting...\n");

    Runtime* runtime = initialize_runtime(runtime_config);
    Device* device = get_device(runtime, 0);
    assert(device);

    img[0] = 69;
    info_print("malloc'd address is: %zu\n", (size_t) img);

    Buffer* buf = import_buffer_host(device, img, sizeof(uint8_t) * WIDTH * HEIGHT * 3);
    uint64_t buf_addr = get_buffer_device_pointer(buf);

    info_print("Device-side address is: %zu\n", buf_addr);

    Program* program = load_program_from_disk(runtime, path);

    wait_completion(launch_kernel(program, device, "aobench_kernel", 16, 16, 1, 1, (void*[]) { &buf_addr }));

    info_print("data %d\n", (int) img[0]);

    destroy_buffer(buf);

    shutdown_runtime(runtime);
}

int main(int argc, char **argv) {
    set_log_level(INFO);
    CompilerConfig compiler_config = default_compiler_config();

    parse_common_args(&argc, argv);
    parse_compiler_config_args(&compiler_config, &argc, argv);

    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    render_host(img, WIDTH, HEIGHT, NSUBSAMPLES);
    saveppm("reference.ppm", WIDTH, HEIGHT, img);

#ifdef ENABLE_ISPC
    render_ispc(img, WIDTH, HEIGHT, NSUBSAMPLES);
    saveppm("ispc.ppm", WIDTH, HEIGHT, img);
#endif

    // render_device(img, WIDTH, HEIGHT, NSUBSAMPLES, "./ao.cl.spv");
    // saveppm("device.ppm", WIDTH, HEIGHT, img);

    free(img);

    return 0;
}
