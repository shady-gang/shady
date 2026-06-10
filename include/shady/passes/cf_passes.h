#ifndef SHADY_CONTROL_FLOW_PASSES_H
#define SHADY_CONTROL_FLOW_PASSES_H

#include "shady/pass.h"

/// Maintains loop-closed SSA form
SHADY_DECLARE_REWRITE_PASS(shd_pass_lcssa)

/// Makes every block top-level by passing all values explicitly
SHADY_DECLARE_REWRITE_PASS(shd_pass_lift_everything)

/// Eliminates critical edges by adding a dummy block at every jump
SHADY_DECLARE_REWRITE_PASS(shd_pass_remove_critical_edges)

#endif
