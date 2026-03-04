#include "shady/driver.h"
#include "shady/ir.h"
#include "shady/cli.h"

#include "util.h"
#include "log.h"

#include <stdlib.h>

static ShdIntSize parse_int_size(String argv) {
    if (strcmp(argv, "8") == 0)
        return ShdIntSize8;
    if (strcmp(argv, "16") == 0)
        return ShdIntSize16;
    if (strcmp(argv, "32") == 0)
        return ShdIntSize32;
    if (strcmp(argv, "64") == 0)
        return ShdIntSize64;
    shd_error("Valid pointer sizes are 8, 16, 32 or 64.");
}

#define TARGET_CONFIG_TOGGLE_OPTIONS(F) \
F(target->ptr_model.address_spaces[AsGeneric].allowed, native-generic-pointers) \
F(target->capabilities.maximal_reconvergence, maximal-reconvergence) \

void shd_parse_target_args(TargetConfig* target, int* pargc, char** argv) {
    int argc = *pargc;

    if (shd_parse_help(pargc, argv, false)) {
        shd_error_print("  --entry-point <foo>                       Selects an entry point for the program to be specialized on.\n");
        shd_error_print("  --word-size <8|16|32|64>                  Sets the word size for physical memory emulation (default=32)\n");
        shd_error_print("  --pointer-size <8|16|32|64>               Sets the pointer size for physical pointers (default=64)\n");
        shd_error_print("  --subgroup-size N                         Sets the subgroup size the program will be specialized for.\n");
        shd_error_print("  --use-native-tailcalls                    Sets the subgroup size the program will be specialized for.\n");
        shd_error_print("  --use-native-fncalls                      Sets the subgroup size the program will be specialized for.\n");
        // TODO: list targets
    }

    // First parse the target, if one is selected
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--target") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                goto invalid_target;
            else if (strcmp(argv[i], "c") == 0)
                target->arch = TgtC;
            else if (strcmp(argv[i], "spirv") == 0)
                target->arch = TgtSPV;
            else if (strcmp(argv[i], "glsl") == 0)
                target->arch = TgtGLSL;
            else if (strcmp(argv[i], "ispc") == 0)
                target->arch = TgtISPC;
            else if (strcmp(argv[i], "cuda") == 0)
                target->arch = TgtCUDA;
            else if (strcmp(argv[i], "none") == 0)
                target->arch = TgtNone;
            else
                goto invalid_target;
            argv[i] = NULL;
            continue;
            invalid_target:
            shd_error_print("--target must be followed with a valid target (see help for list of targets)");
            exit(ShdInvalidTarget);
        }
    }

    // once we've picked a target, we can give defaults to the fine-grained options
    shd_target_configure_defaults_for_arch(target);

    for (int i = 1; i < argc; i++) {
        if (!argv[i])
            continue;

        TARGET_CONFIG_TOGGLE_OPTIONS(PARSE_TOGGLE_OPTION)

        if (strcmp(argv[i], "--subgroup-size") == 0) {
            argv[i] = NULL;
            i++;
            if (i == argc)
                shd_error("Missing subgroup size");
            target->subgroup_size = atoi(argv[i]);
        } /*else if (strcmp(argv[i], "--word-size") == 0) {
            argv[i] = NULL;
            i++;
            target->word_size = parse_int_size(argv[i]);
        } */else if (strcmp(argv[i], "--pointer-size") == 0) {
            argv[i] = NULL;
            i++;
            target->ptr_model.ptr_size = parse_int_size(argv[i]);
        } else if (strcmp(argv[i], "--no-bda") == 0) {
            target->ptr_model.address_spaces[AsGlobal].allowed = false;
        } else if (strcmp(argv[i], "--allow-linkage") == 0) {
            target->capabilities.linkage = true;
        } else if (strcmp(argv[i], "--use-native-tailcalls") == 0) {
            target->capabilities.native_tailcalls = true;
            target->fn_ptr_size = ShdIntSize64;
        } else if (strcmp(argv[i], "--use-native-fncalls") == 0) {
            target->capabilities.native_fncalls = true;
            target->fn_ptr_size = ShdIntSize64;
        } else if (strcmp(argv[i], "--force-memory-emulation") == 0) {
            target->ptr_model.address_spaces[AsPrivate].physical = false;
            target->ptr_model.address_spaces[AsSubgroup].physical = false;
            target->ptr_model.address_spaces[AsShared].physical = false;
        } else {
            continue;
        }
        argv[i] = NULL;
    }

    shd_pack_remaining_args(pargc, argv);
}
