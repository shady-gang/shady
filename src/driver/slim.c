#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

int main(int argc, char** argv) {
    shd_platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    ArenaConfig aconfig = default_arena_config(&args.config.target);
    IrArena* arena = new_ir_arena(&aconfig);
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    ShadyErrorCodes err = driver_load_source_files(&args, mod);
    if (err)
        exit(err);

    err = driver_compile(&args, mod);
    if (err)
        exit(err);
    shd_info_print("Compilation successful\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
