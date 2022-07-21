#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"
#include "portability.h"

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .use_loop_for_fn_body = true,
        .use_loop_for_fn_calls = true,
    };
}

CompilationResult run_compiler_passes(SHADY_UNUSED CompilerConfig* config, IrArena** arena, const Node** program) {
    *program = bind_program(config, *arena, *arena, *program);
    info_print("Bound program successfully: \n");
    info_node(*program);

    *program = normalize(config, *arena, *arena, *program);
    info_print("Normalized program successfully: \n");
    info_node(*program);

    ArenaConfig aconfig = (ArenaConfig) {
        .check_types = true
    };
    IrArena* typed_arena = new_arena(aconfig);
    *program = infer_program(config, *arena, typed_arena, *program);
    destroy_arena(*arena);
    *arena = typed_arena;
    info_print("Type-checked program successfully: \n");
    info_node(*program);

    aconfig.allow_fold = true;
    *arena = new_arena(aconfig);

    *program = lower_cf_instrs(config, *arena, *arena, *program);
    info_print("After lower_cf_instrs pass: \n");
    info_node(*program);

    *program = lower_callc(config, *arena, *arena, *program);
    info_print("After lower_callc pass: \n");
    info_node(*program);

    *program = lower_callf(config, *arena, *arena, *program);
    info_print("After lower_callf pass: \n");
    info_node(*program);

    *program = lower_stack(config, *arena, *arena, *program);
    info_print("After lower_stack pass: \n");
    info_node(*program);

    *program = lower_physical_ptrs(config, *arena, *arena, *program);
    info_print("After lower_physical_ptrs pass: \n");
    info_node(*program);

    // *program = lower_jumps_loop(config, *arena, *arena, *program);
    // info_print("After lower_jumps_loop pass: \n");
    // info_node(*program);

    return CompilationNoError;
}
