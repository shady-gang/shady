#ifndef SHADY_PASS_H
#define SHADY_PASS_H

#include "shady/ir/arena.h"
#include "shady/ir/module.h"
#include "shady/config.h"
#include "shady/rewrite.h"

typedef Module* (RewritePass)(const CompilerConfig* config, Module* src);
typedef bool (OptPass)(const CompilerConfig* config, Module** m);

void shd_run_pass_impl(const CompilerConfig* config, Module** pmod, IrArena* initial_arena, RewritePass pass, String pass_name);
#define RUN_PASS(pass_name) shd_run_pass_impl(config, pmod, initial_arena, pass_name, #pass_name);

void shd_apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name);
#define APPLY_OPT(pass_name) shd_apply_opt_impl(config, &todo, &m, pass_name, #pass_name);

#endif

