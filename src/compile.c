#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .use_loop_for_fn_body = true
    };
}

CompilationResult run_compiler_passes(CompilerConfig* config, IrArena** arena, const Node** program) {
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

    *program = instr2bb(*arena, *arena, *program);
    info_print("instr2bb pass: \n");
    info_node(*program);
}
