#ifndef SHADY_BODY_BUILDER_H
#define SHADY_BODY_BUILDER_H

#include "shady/ir/base.h"

typedef struct BodyBuilder_ BodyBuilder;

/// Used to build a chain of let
BodyBuilder* begin_body_with_mem(IrArena*, const Node*);
BodyBuilder* begin_block_pure(IrArena*);
BodyBuilder* begin_block_with_side_effects(IrArena*, const Node*);

/// Appends an instruction to the builder, may apply optimisations.
/// If the arena is typed, returns a list of variables bound to the values yielded by that instruction
Nodes bind_instruction(BodyBuilder*, const Node* instruction);
const Node* bind_instruction_single(BodyBuilder*, const Node* instruction);
Nodes bind_instruction_named(BodyBuilder*, const Node* instruction, String const output_names[]);

Nodes deconstruct_composite(IrArena* a, BodyBuilder* bb, const Node* value, size_t outputs_count);

Nodes gen_if(BodyBuilder*, Nodes, const Node*, const Node*, Node*);
Nodes gen_match(BodyBuilder*, Nodes, const Node*, Nodes, Nodes, Node*);
Nodes gen_loop(BodyBuilder*, Nodes, Nodes, Node*);

typedef struct {
    Nodes results;
    Node* case_;
    const Node* jp;
} begin_control_t;
begin_control_t begin_control(BodyBuilder*, Nodes);

typedef struct {
    Nodes results;
    Node* loop_body;
    Nodes params;
    const Node* continue_jp;
    const Node* break_jp;
} begin_loop_helper_t;
begin_loop_helper_t begin_loop_helper(BodyBuilder*, Nodes, Nodes, Nodes);

Nodes gen_control(BodyBuilder*, Nodes, Node*);

const Node* bb_mem(BodyBuilder*);

/// Like append bind_instruction, but you explicitly give it information about any yielded values
/// ! In untyped arenas, you need to call this because we can't guess how many things are returned without typing info !
Nodes bind_instruction_outputs_count(BodyBuilder*, const Node* initial_value, size_t outputs_count);

const Node* finish_body(BodyBuilder*, const Node* terminator);
const Node* finish_body_with_return(BodyBuilder*, Nodes args);
const Node* finish_body_with_unreachable(BodyBuilder*);
const Node* finish_body_with_selection_merge(BodyBuilder*, Nodes args);
const Node* finish_body_with_loop_continue(BodyBuilder*, Nodes args);
const Node* finish_body_with_loop_break(BodyBuilder*, Nodes args);
const Node* finish_body_with_join(BodyBuilder*, const Node* jp, Nodes args);
const Node* finish_body_with_jump(BodyBuilder*, const Node* target, Nodes args);

void cancel_body(BodyBuilder*);

const Node* yield_value_and_wrap_in_block(BodyBuilder*, const Node*);
const Node* yield_values_and_wrap_in_block(BodyBuilder*, Nodes);
const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder*, const Node*);

const Node* yield_values_and_wrap_in_compound_instruction(BodyBuilder*, Nodes);

#endif
