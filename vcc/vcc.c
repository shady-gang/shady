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
    VccConfig vcc_options = vcc_init_config(&args.config);
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_vcc_args(&vcc_options, &argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    if (shd_list_count(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    ArenaConfig aconfig = default_arena_config(&args.config.target);
    IrArena* arena = new_ir_arena(&aconfig);

    vcc_check_clang();

    if (vcc_options.only_run_clang)
        vcc_options.tmp_filename = format_string_new("%s", args.output_filename);
    vcc_run_clang(&vcc_options, shd_list_count(args.input_filenames), shd_read_list(String, args.input_filenames));

    if (!vcc_options.only_run_clang) {
        Module* mod = vcc_parse_back_into_module(&args.config, &vcc_options, "my_module");
        driver_compile(&args, mod);
        destroy_ir_arena(get_module_arena(mod));
    }

    info_print("Done\n");

    destroy_vcc_options(vcc_options);
    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
