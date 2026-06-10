#ifndef SHADY_STACK_PASSES_H
#define SHADY_STACK_PASSES_H

#include "shady/pass.h"

SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_fill)

/// Emulates int64 support (unfinished)
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_int)

/// Lowers switches to a tree of ifs (BROKEN)
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_switch_btree)

/// Lowers vectors to arrays
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_vec_arr)

#endif
