#ifndef SHADY_FNCALL_PASSES_H
#define SHADY_FNCALL_PASSES_H

#include "shady/pass.h"

/// Puts the @Leaf annotation on functions that are guaranteed not to be called indirectly
SHADY_DECLARE_REWRITE_PASS(shd_pass_mark_leaf_functions)

/// Promotes return points of non-static controls to top-level functions
SHADY_DECLARE_REWRITE_PASS(shd_pass_lift_indirect_targets)

/// Lowers calls to control + tailcall
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_callf)

/// Lowers non-static control constructs to use built-in scheduler ops
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_dynamic_control)

/// Lowers scheduler ops to the software scheduler
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_tailcalls)

#endif
