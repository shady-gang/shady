#include "ir_private.h"
#include "shady/driver.h"

#include "passes/passes.h"
#include "analysis/verify.h"

#include "log.h"

#ifdef NDEBUG
#define SHADY_RUN_VERIFY 0
#else
#define SHADY_RUN_VERIFY 1
#endif

void shd_run_pass_impl(const CompilerConfig* config, Module** pmod, RewritePass pass, String pass_name, void* payload) {
    Module* old_mod = NULL;
    old_mod = *pmod;
    *pmod = pass(payload, *pmod);
    (*pmod)->sealed = true;
    shd_debugvv_print("After pass %s: \n", pass_name);
    if (SHADY_RUN_VERIFY)
        shd_verify_module(config, *pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod))
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
    old_mod = *pmod;
    if (config->optimisations.cleanup.after_every_pass)
        *pmod = shd_cleanup(config, *pmod);
    shd_log_module(DEBUGVV, config, *pmod);
    if (SHADY_RUN_VERIFY)
        shd_verify_module(config, *pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod))
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
}

void shd_apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name) {
    bool changed = pass(config, m);
    *todo |= changed;

    if (getenv("SHADY_DUMP_CLEAN_ROUNDS") && changed) {
        shd_log_fmt(DEBUGVV, "%s changed something:\n", pass_name);
        shd_log_module(DEBUGVV, config, *m);
    }
}
