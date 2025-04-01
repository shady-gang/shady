#define EXTERNAL_FN /* not static */

#include "ao.h"
#include "../runner/runner_app_common.h"

#include "shady/runner/runner.h"
#include "shady/driver.h"

#include "portability.h"
#include "log.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct {
    CompilerConfig compiler_config;
    RunnerConfig runtime_config;
    CommonAppArgs common_app_args;
} Args;

void saveppm(const char *fname, int w, int h, TEXEL_T* img) {
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    // fwrite(img, w * h * 3, 1, fp);
    for (size_t i = 0; i < w * h * 3; i++) {
        unsigned char c = img[i];
        fwrite(&c, 1, 1, fp);
    }
    fclose(fp);
}

void render_host(TEXEL_T* img, int w, int h, int nsubsamples) {
    int x, y;
    Scalar* fimg = (Scalar *)malloc(sizeof(Scalar) * w * h * 3);
    memset((void *)fimg, 0, sizeof(Scalar) * w * h * 3);

    uint64_t tsn = shd_get_time_nano();
    Ctx ctx = get_init_context();
    init_scene(&ctx);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            render_pixel(&ctx, x, y, w, h, nsubsamples, img);
        }
    }
    uint64_t tpn = shd_get_time_nano();
    shd_info_print("reference rendering took %d us\n", (tpn - tsn) / 1000);
}

#ifdef ENABLE_ISPC
extern void (aobench_kernel)(uint8_t*);

typedef struct {
    uint32_t x, y, z;
} Vec3u;

extern Vec3u builtin_NumWorkgroups;

void render_ispc(TEXEL_T* img, int w, int h, int nsubsamples) {
    uint64_t tsn = shd_get_time_nano();
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
    uint64_t tpn = shd_get_time_nano();
    shd_info_print("ispc rendering took %d us\n", (tpn - tsn) / 1000);
}
#endif

void render_device(Args* args, TEXEL_T *img, int w, int h, int nsubsamples, String path, bool import_memory) {
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    shd_info_print("Shady checkerboard test starting...\n");

    Runner* runtime = shd_rn_initialize(args->runtime_config);
    Device* device = shd_rn_get_device(runtime, args->common_app_args.device);
    assert(device);

    img[0] = 69;
    shd_info_print("malloc'd address is: %zu\n", (size_t) img);

    Buffer* buf;
    if (import_memory)
        buf = shd_rn_import_buffer_host(device, img, sizeof(*img) * WIDTH * HEIGHT * 3);
    else
        buf = shd_rn_allocate_buffer_device(device, sizeof(*img) * WIDTH * HEIGHT * 3);

    uint64_t buf_addr = shd_rn_get_buffer_device_pointer(buf);

    shd_info_print("Device-side address is: %zu\n", buf_addr);

    Module* m;
    CHECK(shd_driver_load_source_file_from_filename(&args->compiler_config, path, "aobench", &m) == NoError, return);
    Program* program = shd_rn_new_program_from_module(runtime, &args->compiler_config, m);

    // run it twice to compile everything and benefit from caches
    shd_rn_wait_completion(shd_rn_launch_kernel(program, device, "aobench_kernel", WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE, 1, 1, (void* []) { &buf_addr }, NULL));
    uint64_t tsn = shd_get_time_nano();
    uint64_t profiled_gpu_time = 0;
    ExtraKernelOptions extra_kernel_options = {
        .profiled_gpu_time = &profiled_gpu_time
    };
    shd_rn_wait_completion(shd_rn_launch_kernel(program, device, "aobench_kernel", WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE, 1, 1, (void* []) { &buf_addr }, &extra_kernel_options));
    uint64_t tpn = shd_get_time_nano();
    shd_info_print("device rendering took %dus (gpu time: %dus)\n", (tpn - tsn) / 1000, profiled_gpu_time / 1000);

    if (!import_memory)
        shd_rn_copy_from_buffer(buf, 0, img, sizeof(*img) * WIDTH * HEIGHT * 3);
    shd_debug_print("data %d\n", (int) img[0]);
    shd_rn_destroy_buffer(buf);

    shd_rn_shutdown(runtime);
}

int main(int argc, char **argv) {
    shd_log_set_level(INFO);
    Args args = {
        .compiler_config = shd_default_compiler_config(),
        .runtime_config = shd_rn_default_config(),
    };

    args.compiler_config.input_cf.restructure_with_heuristics = true;

    shd_parse_common_args(&argc, argv);
    shd_parse_compiler_config_args(&args.compiler_config, &argc, argv);
    shd_rn_cli_parse_config(&args.runtime_config, &argc, argv);
    cli_parse_common_app_arguments(&args.common_app_args, &argc, argv);

    bool do_host = false, do_ispc = false, do_device = false, do_all = true;
    for (size_t i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--only-device") == 0) {
            do_device = true;
            do_all = false;
        } else if (strcmp(argv[i], "--only-host") == 0) {
            do_host = true;
            do_all = false;
        } else if (strcmp(argv[i], "--only-ispc") == 0) {
            do_ispc = true;
            do_all = false;
        } else {
            shd_error_print("Unrecognised argument: %s\n", argv[i]);
            shd_error_die();
        }
    }

    void *img = malloc(WIDTH * HEIGHT * 3 * sizeof(TEXEL_T));

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
        render_device(&args, img, WIDTH, HEIGHT, NSUBSAMPLES, "./ao.comp.c.ll", false);
        saveppm("device.ppm", WIDTH, HEIGHT, img);
    }

    free(img);

    return 0;
}
