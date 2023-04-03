#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "shady/runtime.h"
#include "shady/cli.h"

#include "log.h"
#include "portability.h"
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

#define WIDTH        1024
#define HEIGHT       1024

int
main(int argc, char **argv)
{
    unsigned char *img = (unsigned char *) malloc(WIDTH * HEIGHT * 3);
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
    // parse_runtime_arguments(&argc, argv, &args);
    parse_common_args(&argc, argv);
    parse_compiler_config_args(&compiler_config, &argc, argv);

    info_print("Shady checkerboard test starting...\n");

    Runtime* runtime = initialize_runtime(runtime_config);
    Device* device = get_device(runtime, 0);
    assert(device);

    Buffer* buf = import_buffer_host(device, &img, sizeof(*img));
    uint64_t buf_addr = get_buffer_pointer(buf);

    info_print("Device-side address is: %zu\n", buf_addr);

    Program* program = load_program(runtime, checkerboard_kernel_src);

    uint32_t a0 = 42;
    float a1 = 42.0f;
    wait_completion(launch_kernel(program, device, 1, 1, 1, 2, (void*[]) { &a0, &a1 }));

    shutdown_runtime(runtime);
    saveppm("ao.ppm", WIDTH, HEIGHT, img);
}