#include "shady/ir.h"
#include "shady/driver.h"
#include "shady/print.h"

#include "../shady/passes/passes.h"
#include "../shady/visit.h"

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

    visit_node_operands(v, ~(NcMem | NcDeclaration | NcTerminator), n);
}

static void check_module(Module* mod) {
    Visitor v = { .visit_node_fn = search_for_memstuff };
    visit_module(&v, mod);
    if (expect_memstuff != found_memstuff) {
        error_print("Expected ");
        if (!expect_memstuff)
            error_print("no more ");
        error_print("memory primops in the output.\n");
        dump_module(mod);
        exit(-1);
    }
    dump_module(mod);
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

    cli_pack_remaining_args(pargc, argv);
}

static Module* oracle_passes(const CompilerConfig* config, Module* initial_mod) {
    IrArena* initial_arena = get_module_arena(initial_mod);
    Module** pmod = &initial_mod;

    RUN_PASS(cleanup)
    check_module(*pmod);

    return *pmod;
}

int main(int argc, char** argv) {
    shd_platform_specific_terminal_init_extras();

    DriverConfig args = default_driver_config();
    cli_parse_driver_arguments(&args, &argc, argv);
    cli_parse_common_args(&argc, argv);
    cli_parse_compiler_config_args(&args.config, &argc, argv);
    cli_parse_oracle_args(&argc, argv);
    cli_parse_input_files(args.input_filenames, &argc, argv);

    ArenaConfig aconfig = default_arena_config(&args.config.target);
    aconfig.optimisations.weaken_non_leaking_allocas = true;
    IrArena* arena = new_ir_arena(&aconfig);
    Module* mod = new_module(arena, "my_module"); // TODO name module after first filename, or perhaps the last one

    ShadyErrorCodes err = driver_load_source_files(&args, mod);
    if (err)
        exit(err);

    Module* mod2 = oracle_passes(&args.config, mod);
    destroy_ir_arena(get_module_arena(mod2));

    if (err)
        exit(err);
    info_print("Compilation successful\n");

    destroy_ir_arena(arena);
    destroy_driver_config(&args);
}