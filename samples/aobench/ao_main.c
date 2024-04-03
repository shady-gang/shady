#include "ao.h"
#include "../runtime/runtime_app_common.h"

#include "shady/runtime.h"
#include "shady/driver.h"

#include "log.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

typedef struct {
    CompilerConfig compiler_config;
    RuntimeConfig runtime_config;
    CommonAppArgs common_app_args;
} Args;

static uint64_t timespec_to_nano(struct timespec t) {
    return t.tv_sec * 1000000000 + t.tv_nsec;
}

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

    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t tsn = timespec_to_nano(ts);
    Ctx ctx = get_init_context();
    init_scene(&ctx);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            render_pixel(&ctx, x, y, w, h, nsubsamples, img);
        }
    }
    struct timespec tp;
    timespec_get(&tp, TIME_UTC);
    uint64_t tpn = timespec_to_nano(tp);
    info_print("reference rendering took %d us\n", (tpn - tsn) / 1000);
}

#ifdef ENABLE_ISPC
extern void (aobench_kernel)(uint8_t*);

typedef struct {
    uint32_t x, y, z;
} Vec3u;

extern Vec3u builtin_NumWorkgroups;

void render_ispc(unsigned char *img, int w, int h, int nsubsamples) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t tsn = timespec_to_nano(ts);
    Ctx ctx = get_init_context();
    init_scene(&ctx);
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    builtin_NumWorkgroups.x = WIDTH / 16;
    builtin_NumWorkgroups.y = HEIGHT / 16;
    builtin_NumWorkgroups.z = 1;

    aobench_kernel(img);
    struct timespec tp;
    timespec_get(&tp, TIME_UTC);
    uint64_t tpn = timespec_to_nano(tp);
    info_print("ispc rendering took %d us\n", (tpn - tsn) / 1000);
}
#endif

void render_device(Args* args, unsigned char *img, int w, int h, int nsubsamples, String path) {
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    info_print("Shady checkerboard test starting...\n");

    Runtime* runtime = initialize_runtime(args->runtime_config);
    Device* device = get_device(runtime, args->common_app_args.device);
    assert(device);

    img[0] = 69;
    info_print("malloc'd address is: %zu\n", (size_t) img);

    Buffer* buf = import_buffer_host(device, img, sizeof(uint8_t) * WIDTH * HEIGHT * 3);
    uint64_t buf_addr = get_buffer_device_pointer(buf);

    info_print("Device-side address is: %zu\n", buf_addr);

    Program* program = load_program_from_disk(runtime, &args->compiler_config, path);

    // run it twice to compile everything and benefit from caches
    wait_completion(launch_kernel(program, device, "aobench_kernel", WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE, 1, 1, (void*[]) { &buf_addr }));
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t tsn = timespec_to_nano(ts);
    wait_completion(launch_kernel(program, device, "aobench_kernel", WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE, 1, 1, (void*[]) { &buf_addr }));
    struct timespec tp;
    timespec_get(&tp, TIME_UTC);
    uint64_t tpn = timespec_to_nano(tp);
    info_print("device rendering took %d us\n", (tpn - tsn) / 1000);

    debug_print("data %d\n", (int) img[0]);

    destroy_buffer(buf);

    shutdown_runtime(runtime);
}

int main(int argc, char **argv) {
    set_log_level(INFO);
    Args args = {
        .compiler_config = default_compiler_config(),
        .runtime_config = {
            .use_validation = true,
            .dump_spv = true,
        }
    };
    CompilerConfig compiler_config = default_compiler_config();

    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&compiler_config, &argc, argv);
    cli_parse_common_app_arguments(&args.common_app_args, &argc, argv);

    bool do_host = false, do_ispc = false, do_device = false, do_all = true;
    for (size_t i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--only-device") == 0) {
            do_device = true;
            do_all = false;
        } else if (strcmp(argv[i], "--only-host") == 0) {
            do_host = true;
            do_all = false;
        } else if (strcmp(argv[i], "--only-ispc") == 0) {
            do_ispc = true;
            do_all = false;
        }
    }

    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    if (do_host || do_all) {
        render_host(img, WIDTH, HEIGHT, NSUBSAMPLES);
        saveppm("reference.ppm", WIDTH, HEIGHT, img);
    }

#ifdef ENABLE_ISPC
    if (do_ispc || do_all) {
        render_ispc(img, WIDTH, HEIGHT, NSUBSAMPLES);
        saveppm("ispc.ppm", WIDTH, HEIGHT, img);
    }
#endif

    if (do_device || do_all) {
        render_device(&args, img, WIDTH, HEIGHT, NSUBSAMPLES, "./ao.comp.c.ll");
        saveppm("device.ppm", WIDTH, HEIGHT, img);
    }

    free(img);

    return 0;
}
