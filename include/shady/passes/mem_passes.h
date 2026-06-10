#ifndef SHADY_MEMORY_PASSES_H
#define SHADY_MEMORY_PASSES_H

#include "shady/pass.h"

SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_memcpy)

/// Lowers SizeOf, AlignOf and OffsetOf to constants
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_memory_layout)

/// Lowers physical memory to arrays if needed
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_physical_memory, const PtrModel*, ShdExecutionModel)

#endif
