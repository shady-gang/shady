#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "shady/runtime.h"
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

    RuntimeConfig runtime_config = default_runtime_config();

    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&compiler_config, &argc, argv);
    cli_parse_runtime_config(&runtime_config, &argc, argv);

    shd_info_print("Shady checkerboard test starting...\n");

    Runtime* runtime = initialize_runtime(runtime_config);
    Device* device = get_device(runtime, 0);
    assert(device);

    img[0] = 69;
    shd_info_print("malloc'd address is: %zu\n", (size_t) img);

    int buf_size = sizeof(uint8_t) * WIDTH * HEIGHT * 3;
    Buffer* buf = allocate_buffer_device(device, buf_size);
    copy_to_buffer(buf, 0, img, buf_size);
    uint64_t buf_addr = get_buffer_device_pointer(buf);

    shd_info_print("Device-side address is: %zu\n", buf_addr);

    ArenaConfig aconfig = shd_default_arena_config(&compiler_config.target);
    IrArena* a = new_ir_arena(&aconfig);
    Module* m;
    if (driver_load_source_file(&compiler_config, SrcSlim, sizeof(checkerboard_kernel_src), checkerboard_kernel_src, "checkerboard", &m) != NoError)
        shd_error("Failed to load checkerboard module");
    Program* program = new_program_from_module(runtime, &compiler_config, m);

    wait_completion(launch_kernel(program, device, "checkerboard", 16, 16, 1, 1, (void*[]) { &buf_addr }, NULL));

    copy_from_buffer(buf, 0, img, buf_size);
    shd_info_print("data %d\n", (int) img[0]);

    destroy_buffer(buf);

    shutdown_runtime(runtime);
    saveppm("checkerboard.ppm", WIDTH, HEIGHT, img);
    destroy_ir_arena(a);
    free(img);
}
