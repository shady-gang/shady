#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

int main(int argc, char** argv) {
    shd_platform_specific_terminal_init_extras();

    bool help = shd_parse_help(&argc, argv, false);
    if (help) {
        shd_error_print("Usage: slim source.slim [arguments]\n");
        shd_error_print("Available arguments: \n");
    }

    DriverConfig args = shd_default_driver_config();
    shd_parse_common_args(&argc, argv);
    shd_parse_compiler_config_args(&args.config, &argc, argv);
    shd_parse_driver_args(&args, &argc, argv);

    // Obtain the default target, default the arch to the filename but allow overriding it when parsing target args
    TargetConfig target_config = shd_default_target_config();
    target_config.arch = shd_driver_guess_target_through_name(args.output_filename);
    shd_parse_target_args(&target_config, &argc, argv);

    shd_driver_configure_from_target(&args, &target_config);

    ShaderLoweringConfig lowering_config = shd_default_shader_target_config();
    shd_parse_shader_target_config_args(&lowering_config, &argc, argv);

    if (help)
        exit(0);

    shd_parse_help(&argc, argv, true);
    shd_driver_parse_input_files(args.input_filenames, &argc, argv);

    MachineRules rules = get_machine_rules_from_target_config(&target_config);
    ArenaConfig aconfig = shd_default_arena_config(&rules);
    IrArena* arena = shd_new_ir_arena(&aconfig);
    Module* mod = shd_new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    ShadyErrorCodes err = shd_driver_load_source_files(&args.config, &target_config, args.input_filenames, mod);
    if (err)
        exit(err);

    err = shd_driver_compile(&args, target_config.arch != TgtNone ? &lowering_config : NULL, target_config, mod);
    if (err)
        exit(err);
    shd_info_print("Compilation successful\n");

    shd_destroy_ir_arena(arena);
    shd_destroy_driver_config(&args);
}
