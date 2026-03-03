#ifndef SHADY_SPIRV_PASSES_H
#define SHADY_SPIRV_PASSES_H

#include "shady/pass.h"

SHADY_DECLARE_REWRITE_PASS(shd_lower_to_callable_shaders)

/// Avoids some implementation bugs
SHADY_DECLARE_REWRITE_PASS(shd_spvbe_pass_remove_bda_params)

/// Makes sure to only use explicit-layout structs where allowed
SHADY_DECLARE_REWRITE_PASS(shd_spvbe_pass_specialize_explicit_layout)
SHADY_DECLARE_REWRITE_PASS(shd_spv_lower_entrypoint_args)

#endif
