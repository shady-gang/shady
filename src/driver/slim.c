#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <assert.h>

#ifndef HOOK_STUFF
#define HOOK_STUFF
#endif

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    HOOK_STUFF
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    IrArena* arena = new_ir_arena(default_arena_config());
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    driver_load_source_files(&args, mod);

    driver_compile(&args, mod);
    info_print("Done\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
