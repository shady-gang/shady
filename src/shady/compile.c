#include "ir_private.h"
#include "shady/driver.h"
#include "shady/pass.h"

#include "analysis/verify.h"

#include "log.h"
#include "util.h"

#ifdef NDEBUG
#define SHADY_RUN_VERIFY 0
#else
#define SHADY_RUN_VERIFY 1
#endif

SHADY_DECLARE_REWRITE_PASS(shd_cleanup)

Module* shd_before_pass_impl(const CompilerConfig* config, Module** pmod, String pass_name) {
    // if (shd_string_starts_with(pass_name, "shd_pass_"))
    //     pass_name = &pass_name[strlen("shd_pass_")];
    // String log_pass_e = getenv("SHADY_LOG_PASS");
    // bool log_pass = log_pass_e && (shd_string_starts_with(log_pass_e, "1"));
    // if (log_pass_e)
    //     shd_configure_bool_flag_in_list(log_pass_e, pass_name, &log_pass);
    // if (log_pass)
    //     shd_log_fmt(INFO, "After pass %s: \n", pass_name);

    Module* old_mod = NULL;
    old_mod = *pmod;
    return old_mod;
}

void shd_after_pass_impl(const CompilerConfig* config, Module** pmod, String pass_name, Module* old_mod) {
    if (shd_string_starts_with(pass_name, "shd_pass_"))
        pass_name = &pass_name[strlen("shd_pass_")];
    String log_pass_e = getenv("SHADY_LOG_PASS");
    bool log_pass = log_pass_e && (shd_string_starts_with(log_pass_e, "1"));
    if (log_pass_e)
        shd_configure_bool_flag_in_list(log_pass_e, pass_name, &log_pass);
    if (log_pass)
        shd_log_fmt(INFO, "After pass %s: \n", pass_name);

    (*pmod)->sealed = true;
    if (SHADY_RUN_VERIFY)
        shd_verify_module(*pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod))
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
    old_mod = *pmod;
    if (config->optimisations.cleanup.after_every_pass)
        *pmod = shd_cleanup(config, *pmod);
    if (log_pass)
        shd_log_module(INFO, *pmod);
    if (SHADY_RUN_VERIFY)
        shd_verify_module(*pmod);
    if (shd_module_get_arena(old_mod) != shd_module_get_arena(*pmod))
        shd_destroy_ir_arena(shd_module_get_arena(old_mod));
}

void shd_apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name) {
    bool changed = pass(config, m);
    *todo |= changed;

    if (getenv("SHADY_DUMP_CLEAN_ROUNDS") && changed) {
        shd_log_fmt(DEBUGVV, "%s changed something:\n", pass_name);
        shd_log_module(DEBUGVV, *m);
    }
}
