#ifndef SHADY_SLIM_PASSES_H
#define SHADY_SLIM_PASSES_H

#include "shady/pass.h"

/// Removes all Unresolved nodes and replaces them with the appropriate decl/value
SHADY_DECLARE_REWRITE_PASS(slim_pass_bind);

/// Enforces the grammar, notably by let-binding any intermediary result
SHADY_DECLARE_REWRITE_PASS(slim_pass_normalize);

/// Makes sure every node is well-typed
SHADY_DECLARE_REWRITE_PASS(slim_pass_infer);

#endif
