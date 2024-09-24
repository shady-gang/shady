#ifndef SHADY_PASS_H
#define SHADY_PASS_H

#include "shady/config.h"
#include "rewrite.h"

typedef Module* (RewritePass)(const CompilerConfig* config, Module* src);
typedef bool (OptPass)(const CompilerConfig* config, Module** m);

void run_pass_impl(const CompilerConfig* config, Module** pmod, IrArena* initial_arena, RewritePass pass, String pass_name);
#define RUN_PASS(pass_name) run_pass_impl(config, pmod, initial_arena, pass_name, #pass_name);

void apply_opt_impl(const CompilerConfig* config, bool* todo, Module** m, OptPass pass, String pass_name);
#define APPLY_OPT(pass_name) apply_opt_impl(config, &todo, &m, pass_name, #pass_name);

#endif

