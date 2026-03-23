#ifndef SHADY_GROUP_OPERATIONS_PASSES_H
#define SHADY_GROUP_OPERATIONS_PASSES_H

#include "shady/pass.h"

/// Transforms
/// SpvOpGroupXXX(Scope, 'GroupOperationInclusiveScan', v)
/// into
/// SpvOpGroupXXX(Scope, 'GroupOperationExclusiveScan', v) op v
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_inclusive_scan)

/// Lowers the built-in mask type to the actual mask type for the target
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_mask)

/// Lowers subgroup variables to shared memory
/// Also bans subgroup memory from being used in the module going forward.
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_subgroup_vars, uint32_t)

/// Emulates certain subgroup operations
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_subgroup_ops)

#endif
