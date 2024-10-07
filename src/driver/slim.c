#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

int main(int argc, char** argv) {
    shd_platform_specific_terminal_init_extras();

    DriverConfig args = shd_default_driver_config();
    shd_parse_driver_args(&args, &argc, argv);
    shd_parse_common_args(&argc, argv);
    shd_parse_compiler_config_args(&args.config, &argc, argv);
    shd_driver_parse_input_files(args.input_filenames, &argc, argv);

    ArenaConfig aconfig = shd_default_arena_config(&args.config.target);
    IrArena* arena = shd_new_ir_arena(&aconfig);
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    ShadyErrorCodes err = shd_driver_load_source_files(&args, mod);
    if (err)
        exit(err);

    err = shd_driver_compile(&args, mod);
    if (err)
        exit(err);
    shd_info_print("Compilation successful\n");

    shd_destroy_ir_arena(arena);
    shd_destroy_driver_config(&args);
}
