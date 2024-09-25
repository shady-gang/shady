#include "parser.h"

#include "shady/pass.h"

#include "../shady/transform/internal_constants.h"
#include "../shady/passes/passes.h"

#include "log.h"

/// Removes all Unresolved nodes and replaces them with the appropriate decl/value
RewritePass bind_program;
/// Enforces the grammar, notably by let-binding any intermediary result
RewritePass normalize;
/// Makes sure every node is well-typed
RewritePass infer_program;

void slim_parse_string(ParserConfig config, const char* contents, Module* mod);

Module* parse_slim_module(const CompilerConfig* config, ParserConfig pconfig, const char* contents, String name) {
    ArenaConfig aconfig = default_arena_config(&config->target);
    aconfig.name_bound = false;
    aconfig.check_op_classes = false;
    aconfig.check_types = false;
    aconfig.validate_builtin_types = false;
    aconfig.allow_fold = false;
    IrArena* initial_arena = new_ir_arena(&aconfig);
    Module* m = new_module(initial_arena, name);
    slim_parse_string(pconfig, contents, m);
    Module** pmod = &m;
    Module* old_mod = NULL;

    shd_debugv_print("Parsed slim module:\n");
    shd_log_module(DEBUGV, config, *pmod);

    generate_dummy_constants(config, *pmod);

    RUN_PASS(bind_program)
    RUN_PASS(normalize)

    RUN_PASS(normalize_builtins)
    RUN_PASS(infer_program)
    RUN_PASS(lower_cf_instrs)

    destroy_ir_arena(initial_arena);
    return *pmod;
}
