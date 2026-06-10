#ifndef SHADY_PASS_H
#define SHADY_PASS_H

#include "shady/ir/arena.h"
#include "shady/ir/module.h"
#include "shady/rewrite.h"

typedef struct CompilerConfig_ CompilerConfig;

Module* shd_before_pass_impl(const CompilerConfig* config, Module** pmod, String pass_name);
void shd_after_pass_impl(const CompilerConfig* config, Module** pmod, String pass_name, Module*);

#define SHADY_DECLARE_REWRITE_PASS_(pass_name, prefix, ...) prefix Module* pass_name(const CompilerConfig* config, Module* src, ##__VA_ARGS__);
#define SHADY_DECLARE_REWRITE_PASS(pass_name, ...) SHADY_DECLARE_REWRITE_PASS_(pass_name, , ##__VA_ARGS__)
#define SHADY_DECLARE_REWRITE_PASS_STATIC(pass_name, ...) SHADY_DECLARE_REWRITE_PASS_(pass_name, static, ##__VA_ARGS__)

#define SHADY_APPLY_REWRITE_PASS(pass_name, ...) { \
Module* old_mod = shd_before_pass_impl(config, pmod, #pass_name); \
*pmod = pass_name(config, old_mod, ##__VA_ARGS__); \
shd_after_pass_impl(config, pmod, #pass_name, old_mod); \
}

typedef bool (OptPass)(const CompilerConfig* config, Module** m);
void shd_apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name);
#define APPLY_OPT(pass_name) shd_apply_opt_impl(config, &todo, &m, pass_name, #pass_name);

SHADY_DECLARE_REWRITE_PASS(shd_import)

#endif

