#ifndef SHADY_STACK_PASSES_H
#define SHADY_STACK_PASSES_H

#include "shady/pass.h"

/// Wraps function bodies to save/restore the stack upon entry/exit
SHADY_DECLARE_REWRITE_PASS(shd_pass_setup_stack_frames)

/// Lowers Alloca to stack frames
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_alloca)

/// Adds a stack pointer global variable and lower stack access ops to it
SHADY_DECLARE_REWRITE_PASS(shd_pass_lower_stack_access)

#endif
