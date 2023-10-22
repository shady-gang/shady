#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <assert.h>

int main(int argc, char** argv) {
    platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    parse_driver_arguments(&args, &argc, argv);
    parse_common_args(&argc, argv);
    parse_compiler_config_args(&args.config, &argc, argv);
    parse_input_files(args.input_filenames, &argc, argv);

    IrArena* arena = new_ir_arena(default_arena_config());
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    if (entries_count_list(args.input_filenames) == 0) {
        error_print("Missing input file. See --help for proper usage");
        exit(MissingInputArg);
    }

    size_t num_source_files = entries_count_list(args.input_filenames);
    for (size_t i = 0; i < num_source_files; i++) {
        int err = parse_file_from_filename(read_list(const char*, args.input_filenames)[i], mod);
        if (err)
            return err;
    }

    driver_compile(&args, mod);
    info_print("Done\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}
