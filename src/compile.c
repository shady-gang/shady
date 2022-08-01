#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"
#include "portability.h"
#include "analysis/verify.h"

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .use_loop_for_fn_body = true,
        .use_loop_for_fn_calls = true,
    };
}
#define RUN_PASS(pass_name) \
old_arena = *arena;                                        \
*arena = new_arena(aconfig);                               \
*program = pass_name(config, old_arena, *arena, *program); \
info_print("After "#pass_name" pass: \n");                 \
info_node(*program);                                       \
verify_program(*program);                                  \
destroy_arena(old_arena);

CompilationResult run_compiler_passes(SHADY_UNUSED CompilerConfig* config, IrArena** arena, const Node** program) {
    ArenaConfig aconfig = {
        .allow_fold = false,
        .check_types = false
    };
    IrArena* old_arena;

    RUN_PASS(bind_program)
    RUN_PASS(normalize)

    aconfig.check_types = true;
    RUN_PASS(infer_program)

    aconfig.allow_fold = true;

    RUN_PASS(lower_cf_instrs)
    RUN_PASS(lower_callc)
    RUN_PASS(lower_callf)
    RUN_PASS(lower_tailcalls)
    RUN_PASS(lower_mask)
    RUN_PASS(lower_stack)
    RUN_PASS(lower_physical_ptrs)

    return CompilationNoError;
}
