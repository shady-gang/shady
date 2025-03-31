#ifndef SHADY_PASSES_H

#include "shady/ir.h"
#include "shady/pass.h"

/// @name Boring, regular compiler stuff
/// @{

RewritePass shd_cleanup;

/// @}

/// @name Initial CF lowering passes
/// @{

/// Gets rid of structured control flow constructs, and turns them into branches, joins and tailcalls
RewritePass shd_pass_lower_cf_instrs;
/// Uses shady.scope annotations to insert control blocks
RewritePass shd_pass_scope2control;
RewritePass shd_pass_lift_everything;
RewritePass shd_pass_remove_critical_edges;
RewritePass shd_pass_lcssa;
RewritePass shd_pass_scope_heuristic;
/// Try to identify reconvergence points throughout the program for unstructured control flow programs
RewritePass shd_pass_reconvergence_heuristics;

RewritePass shd_pass_add_init_fini;

/// @}

/// @name Emulation misc.
/// @{

RewritePass shd_pass_lower_vec_arr;
RewritePass shd_pass_lower_workgroups;
RewritePass shd_pass_lower_inclusive_scan;

/// @}

/// @name Optimisation passes
/// @{

/// Eliminates all Constant decls
RewritePass shd_pass_eliminate_constants;

/// Ditto but for @Inline ones only
RewritePass shd_pass_eliminate_inlineable_constants;
/// In addition, also inlines function calls according to heuristics
RewritePass shd_pass_inline;
OptPass shd_opt_mem2reg;

RewritePass shd_pass_restructurize;
RewritePass shd_pass_lower_switch_btree;

RewritePass shd_pass_specialize_entry_point;
RewritePass shd_pass_specialize_execution_model;

/// @}

#define SHADY_PASSES_H

#endif
