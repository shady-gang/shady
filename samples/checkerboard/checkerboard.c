#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "shady/ir/arena.h"
#include "shady/runner.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"

#include "checkerboard_kernel_src.h"

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

#define WIDTH        256
#define HEIGHT       256

int main(int argc, char **argv)
{
    uint8_t* img = malloc(sizeof(uint8_t) * WIDTH * HEIGHT * 3);
    for (size_t i = 0; i < WIDTH; i++) {
        for (size_t j = 0; j < HEIGHT; j++) {
            img[j * WIDTH * 3 + i * 3 + 0] = 255;
            img[j * WIDTH * 3 + i * 3 + 1] = 0;
            img[j * WIDTH * 3 + i * 3 + 2] = 255;
        }
    }

    shd_log_set_level(INFO);
    CompilerConfig compiler_config = shd_default_compiler_config();

    RuntimeConfig runtime_config = shd_rt_default_config();

    shd_parse_common_args(&argc, argv);
    shd_parse_compiler_config_args(&compiler_config, &argc, argv);
    shd_rt_cli_parse_runtime_config(&runtime_config, &argc, argv);

    shd_info_print("Shady checkerboard test starting...\n");

    Runtime* runtime = shd_rt_initialize(runtime_config);
    Device* device = shd_rt_get_device(runtime, 0);
    assert(device);

    img[0] = 69;
    shd_info_print("malloc'd address is: %zu\n", (size_t) img);

    int buf_size = sizeof(uint8_t) * WIDTH * HEIGHT * 3;
    Buffer* buf = shd_rt_allocate_buffer_device(device, buf_size);
    shd_rt_copy_to_buffer(buf, 0, img, buf_size);
    uint64_t buf_addr = shd_rt_get_buffer_device_pointer(buf);

    shd_info_print("Device-side address is: %zu\n", buf_addr);

    ArenaConfig aconfig = shd_default_arena_config(&compiler_config.target);
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* m;
    if (shd_driver_load_source_file(&compiler_config, SrcSlim, sizeof(checkerboard_kernel_src), checkerboard_kernel_src,
                                    "checkerboard", &m) != NoError)
        shd_error("Failed to load checkerboard module");
    Program* program = shd_rt_new_program_from_module(runtime, &compiler_config, m);

    shd_rt_wait_completion(shd_rt_launch_kernel(program, device, "checkerboard", 16, 16, 1, 1, (void* []) { &buf_addr }, NULL));

    shd_rt_copy_from_buffer(buf, 0, img, buf_size);
    shd_info_print("data %d\n", (int) img[0]);

    shd_rt_destroy_buffer(buf);

    shd_rt_shutdown(runtime);
    saveppm("checkerboard.ppm", WIDTH, HEIGHT, img);
    shd_destroy_ir_arena(a);
    free(img);
}
