#ifndef SHADY_STRUCTURED_CONTROL_FLOW_PASSES_H
#define SHADY_STRUCTURED_CONTROL_FLOW_PASSES_H

#include "shady/pass.h"

/// Lowers structured if, switch and loops to control-based SCF
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_cf_instrs)

/// Inserts control constructs in unstructured control flow according to post-dominator and loop tree analysis
SHADY_DECLARE_REWRITE_PASS(shd_pass_reconvergence_heuristics)

/// Inserts control constructs in unstructured control flow according to scope nesting information
SHADY_DECLARE_REWRITE_PASS(shd_pass_scope2control)

/// Inserts scope nesting information using loop tree and post-dominator
SHADY_DECLARE_REWRITE_PASS(shd_pass_scope_heuristic)

/// Lowers static control constructs into loops and if-ladders (conventional SCF fit for GLSL/structured spir-v constraints)
SHADY_DECLARE_REWRITE_PASS(shd_pass_restructurize)

#endif
