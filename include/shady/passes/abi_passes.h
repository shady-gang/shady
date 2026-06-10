#ifndef SHADY_ABI_PASSES_H
#define SHADY_ABI_PASSES_H

#include "shady/pass.h"

typedef struct {
    AddressSpace src_as;
    AddressSpace dst_as;
    bool use_copies;
} Global2LocalsPassConfig;

/// Lowers certain global variables to local variables allocated in the entry point
SHADY_DECLARE_REWRITE_PASS(shd_pass_globals_to_locals, Global2LocalsPassConfig)

/// Lowers certain global variables to kernel parameters
SHADY_DECLARE_REWRITE_PASS(shd_pass_globals_to_params)

/// Emulates workgroups by iterating over the grid
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_workgroups, const TargetConfig*)

/// Assigns an actual address space to generic globals
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_generic_globals, AddressSpace)

/// Adds calls to init and fini arrounds the entry points
SHADY_DECLARE_REWRITE_PASS(shd_pass_call_init_fini)

#endif
