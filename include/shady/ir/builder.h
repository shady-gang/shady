#ifndef SHADY_BUILDER_H
#define SHADY_BUILDER_H

#include "shady/ir/base.h"

typedef struct BodyBuilder_ BodyBuilder;

/// Used to build a chain of let
BodyBuilder* shd_bld_begin(IrArena* a, const Node* mem);
BodyBuilder* shd_bld_begin_pure(IrArena* a);
BodyBuilder* shd_bld_begin_pseudo_instr(IrArena* a, const Node* mem);

IrArena* shd_get_bb_arena(BodyBuilder* bb);

/// Appends an instruction to the builder, may apply optimisations.
/// If the arena is typed, returns a list of variables bound to the values yielded by that instruction
Nodes shd_bld_add_instruction_extract(BodyBuilder* bb, const Node* instruction);
const Node* shd_bld_add_instruction(BodyBuilder* bb, const Node* instr);

/// Like append shd_bld_add_instruction_extract, but you explicitly give it information about any yielded values
/// ! In untyped arenas, you need to call this because we can't guess how many things are returned without typing info !
Nodes shd_bld_add_instruction_extract_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count);

Nodes shd_bld_if(BodyBuilder* bb, Nodes yield_types, const Node* condition, const Node* true_case, Node* false_case);
Nodes shd_bld_match(BodyBuilder* bb, Nodes yield_types, const Node* inspectee, Nodes literals, Nodes cases, Node* default_case);
Nodes shd_bld_loop(BodyBuilder* bb, Nodes yield_types, Nodes initial_args, Node* body);

typedef struct {
    Nodes results;
    Node* case_;
    const Node* jp;
} begin_control_t;
begin_control_t shd_bld_begin_control(BodyBuilder* bb, Nodes yield_types);

typedef struct {
    Nodes results;
    Node* loop_body;
    Nodes params;
    const Node* continue_jp;
    const Node* break_jp;
} begin_loop_helper_t;
begin_loop_helper_t shd_bld_begin_loop_helper(BodyBuilder* bb, Nodes yield_types, Nodes arg_types, Nodes initial_values);

Nodes shd_bld_control(BodyBuilder* bb, Nodes yield_types, Node* body);

const Node* shd_bld_mem(BodyBuilder* bb);

const Node* shd_bld_finish(BodyBuilder* bb, const Node* terminator);
const Node* shd_bld_return(BodyBuilder* bb, Nodes args);
const Node* shd_bld_unreachable(BodyBuilder* bb);
const Node* shd_bld_selection_merge(BodyBuilder* bb, Nodes args);
const Node* shd_bld_loop_continue(BodyBuilder* bb, Nodes args);
const Node* shd_bld_loop_break(BodyBuilder* bb, Nodes args);
const Node* shd_bld_join(BodyBuilder* bb, const Node* jp, Nodes args);
const Node* shd_bld_jump(BodyBuilder* bb, const Node* target, Nodes args);
const Node* shd_bld_indirect_tail_call(BodyBuilder* bb, const Node* target, Nodes args);

void shd_bld_cancel(BodyBuilder* bb);

const Node* shd_bld_to_instr_yield_value(BodyBuilder* bb, const Node* value);
const Node* shd_bld_to_instr_yield_values(BodyBuilder* bb, Nodes values);
const Node* shd_bld_to_instr_with_last_instr(BodyBuilder* bb, const Node* instruction);

const Node* shd_bld_to_instr_pure_with_values(BodyBuilder* bb, Nodes values);

typedef struct Rewriter_ Rewriter;
BodyBuilder* shd_bld_begin_fn_rewrite(Rewriter* r, const Node* old, Node** new_);
void shd_bld_finish_fn_rewrite(Rewriter* r, const Node* old, Node* new_, BodyBuilder* bld);

#endif
