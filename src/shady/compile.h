#ifndef SHADY_COMPILE_H
#define SHADY_COMPILE_H

#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"
#include "analysis/verify.h"

#ifdef NDEBUG
#define SHADY_RUN_VERIFY 0
#else
#define SHADY_RUN_VERIFY 1
#endif

void run_pass_impl(const CompilerConfig* config, Module** pmod, IrArena* initial_arena, RewritePass pass, String pass_name);

#define RUN_PASS(pass_name) run_pass_impl(config, pmod, initial_arena, pass_name, #pass_name);

#endif
