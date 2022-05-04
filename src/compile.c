#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .use_loop_for_fn_body = true,
        .use_loop_for_fn_calls = true,
    };
}

CompilationResult run_compiler_passes(__attribute__ ((unused)) CompilerConfig* config, IrArena** arena, const Node** program) {
    *program = bind_program(*arena, *arena, *program);
    info_print("Bound program successfully: \n");
    info_node(*program);

    IrArena* typed_arena = new_arena((IrConfig) {
        .check_types = true
    });
    *program = type_program(*arena, typed_arena, *program);
    destroy_arena(*arena);
    *arena = typed_arena;
    info_print("Typed program successfully: \n");
    info_node(*program);

    *program = lower_cf_instrs(*arena, *arena, *program);
    info_print("After lower_cf_instrs pass: \n");
    info_node(*program);

    *program = lower_callc(*arena, *arena, *program);
    info_print("After lower_callc pass: \n");
    info_node(*program);

    *program = lower_callf(*arena, *arena, *program);
    info_print("After lower_callf pass: \n");
    info_node(*program);

    *program = lower_stack(*arena, *arena, *program);
    info_print("After lower_stack pass: \n");
    info_node(*program);

    return CompilationNoError;
}
