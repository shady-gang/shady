#include "shady/ir.h"
#include "vcc/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "growy.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    VccConfig vcc_options = vcc_init_config();
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_vcc_args(&vcc_options, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    if (entries_count_list(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    ArenaConfig aconfig = default_arena_config();
    IrArena* arena = new_ir_arena(aconfig);
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    vcc_check_clang();

    if (vcc_options.only_run_clang)
        vcc_options.tmp_filename = format_string_new("%s", args.output_filename);
    vcc_run_clang(&vcc_options, entries_count_list(args.input_filenames), read_list(String, args.input_filenames));

    if (!vcc_options.only_run_clang) {
        vcc_parse_back_into_module(&vcc_options, mod);
        driver_compile(&args, mod);
    }

    info_print("Done\n");

    destroy_vcc_options(vcc_options);
    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
