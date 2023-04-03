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

#define RUN_PASS(pass_name)                             \
old_mod = mod;                                          \
old_arena = tmp_arena;                                  \
tmp_arena = new_ir_arena(aconfig);                      \
mod = new_module(tmp_arena, get_module_name(old_mod));  \
pass_name(config, old_mod, mod);                        \
debug_print("After "#pass_name" pass: \n");             \
log_module(DEBUG, config, mod);                         \
if (SHADY_RUN_VERIFY)                                   \
verify_module(mod);                                     \
mod->sealed = true;                                     \
if (old_arena) destroy_ir_arena(old_arena);

#endif
