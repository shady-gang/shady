#ifndef SHADY_OPT_PASSES_H
#define SHADY_OPT_PASSES_H

#include "shady/pass.h"

/// Applies various cleanup optimizations until a fixed-point is reached
SHADY_DECLARE_REWRITE_PASS(shd_cleanup)

/// Eliminates constants,
SHADY_DECLARE_REWRITE_PASS(shd_pass_eliminate_constants, bool)

SHADY_DECLARE_REWRITE_PASS(shd_pass_inline)

#endif
