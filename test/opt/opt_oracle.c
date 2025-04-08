#include "shady/ir.h"
#include "shady/driver.h"
#include "shady/print.h"
#include "shady/visit.h"

#include "../shady/passes/passes.h"

#include "log.h"
#include "portability.h"

#include <string.h>
#include <assert.h>
#include <stdlib.h>

static bool expect_memstuff = false;
static bool found_memstuff = false;

static void search_for_memstuff(Visitor* v, const Node* n) {
    switch (n->tag) {
        case Load_TAG:
        case Store_TAG:
        case CopyBytes_TAG:
        case FillBytes_TAG:
        case StackAlloc_TAG:
        case LocalAlloc_TAG: {
            found_memstuff = true;
            break;
        }
        default: break;
    }

    shd_visit_node_operands(v, ~(NcMem | NcFunction | NcTerminator), n);
}

static void check_module(Module* mod) {
    Visitor v = { .visit_node_fn = search_for_memstuff };
    shd_visit_module(&v, mod);
    if (expect_memstuff != found_memstuff) {
        shd_error_print("Expected ");
        if (!expect_memstuff)
            shd_error_print("no more ");
        shd_error_print("memory primops in the output.\n");
        shd_dump_module(mod);
        exit(-1);
    }
    shd_dump_module(mod);
    exit(0);
}

static void cli_parse_oracle_args(int* pargc, char** argv) {
    int argc = *pargc;

    for (int i = 1; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        else if (strcmp(argv[i], "--expect-memops") == 0) {
            argv[i] = NULL;
            expect_memstuff = true;
            continue;
        }
    }

    shd_pack_remaining_args(pargc, argv);
}

static Module* oracle_passes(const CompilerConfig* config, Module* initial_mod) {
    IrArena* initial_arena = shd_module_get_arena(initial_mod);
    Module** pmod = &initial_mod;

    RUN_PASS(shd_cleanup, config)
    check_module(*pmod);

    return *pmod;
}

int main(int argc, char** argv) {
    shd_platform_specific_terminal_init_extras();

    DriverConfig args = shd_default_driver_config();
    shd_parse_common_args(&argc, argv);
    shd_parse_compiler_config_args(&args.config, &argc, argv);

    shd_parse_driver_args(&args, &argc, argv);

    TargetConfig target_config = shd_default_target_config();
    shd_driver_configure_target(&target_config, &args);
    shd_parse_target_args(&target_config, &argc, argv);

    cli_parse_oracle_args(&argc, argv);
    shd_driver_parse_input_files(args.input_filenames, &argc, argv);

    ArenaConfig aconfig = shd_default_arena_config(&target_config);
    aconfig.optimisations.weaken_non_leaking_allocas = true;
    IrArena* arena = shd_new_ir_arena(&aconfig);
    Module* mod = shd_new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    ShadyErrorCodes err = shd_driver_load_source_files(&args.config, &target_config, args.input_filenames, mod);
    if (err)
        exit(err);

    Module* mod2 = oracle_passes(&args.config, mod);
    shd_destroy_ir_arena(shd_module_get_arena(mod2));

    if (err)
        exit(err);
    shd_info_print("Compilation successful\n");

    shd_destroy_ir_arena(arena);
    shd_destroy_driver_config(&args);
}