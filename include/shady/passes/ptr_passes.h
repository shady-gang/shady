#ifndef SHADY_POINTER_PASSES_H
#define SHADY_POINTER_PASSES_H

#include "shady/pass.h"

/// Lowers pointers-to-arrays to plain pointers
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_decay_ptrs)

/// Lowers generic pointers to tagged unsigned ints
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_generic_ptrs)

/// Lowers _all_ ptr arithmetic ops to unsigned int math
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_lea, const PtrModel*)

/// Lowers ptr arithmetic ops to unsigned int math
/// If 'always' is false, only ptr arithmetic whose base ptr was rewritten to unsigned ints are lowered
const Node* shd_lower_lea_helper(Rewriter*, const Node*, bool always);

SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_nullptr)

/// Hail mary attempt to lower physical pointers to logical by turning casts into LEAs
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_logical_pointers)

#endif
