#include "ir_private.h"
#include "log.h"
#include "portability.h"
#include "type.h"
#include "transform/ir_gen_helpers.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    Structured_constructTag tag;
    union NodesUnion payload;
} BlockEntry;

typedef struct {
    BlockEntry structured;
    Nodes vars;
} StackEntry;

BodyBuilder* begin_body_with_mem(IrArena* a, const Node* mem) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .stack = new_list(StackEntry),
        .mem = mem,
    };
    return bb;
}

BodyBuilder* begin_block_with_side_effects(IrArena* a, const Node* mem) {
    Node* block = basic_block(a, empty(a), NULL);
    BodyBuilder* builder = begin_body_with_mem(a, get_abstraction_mem(block));
    builder->tail_block = block;
    builder->block_entry_block = block;
    builder->block_entry_mem = mem;
    return builder;
}

BodyBuilder* begin_block_pure(IrArena* a) {
    BodyBuilder* builder = begin_body_with_mem(a, NULL);
    return builder;
}

const Node* bb_mem(BodyBuilder* bb) {
    return bb->mem;
}

Nodes deconstruct_composite(IrArena* a, BodyBuilder* bb, const Node* value, size_t outputs_count) {
    if (outputs_count > 1) {
        LARRAY(const Node*, extracted, outputs_count);
        for (size_t i = 0; i < outputs_count; i++)
            extracted[i] = gen_extract_single(bb, value, int32_literal(bb->arena, i));
        return nodes(bb->arena, outputs_count, extracted);
    } else if (outputs_count == 1)
        return singleton(value);
    else
        return empty(bb->arena);
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, size_t outputs_count) {
    if (bb->arena->config.check_types) {
        assert(is_mem(instruction));
    }
    if (is_mem(instruction) && /* avoid things like ExtInstr with null mem input! */ get_parent_mem(instruction))
        bb->mem = instruction;
    return deconstruct_composite(bb->arena, bb, instruction, outputs_count);
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, singleton(instruction->type).count);
}

const Node* bind_instruction_single(BodyBuilder* bb, const Node* instr) {
    return first(bind_instruction_outputs_count(bb, instr, 1));
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    assert(output_names);
    return bind_internal(bb, instruction, singleton(instruction->type).count);
}

Nodes bind_instruction_outputs_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count) {
    return bind_internal(bb, instruction, outputs_count);
}

static const Node* build_body(BodyBuilder* bb, const Node* terminator) {
    IrArena* a = bb->arena;
    size_t stack_size = entries_count_list(bb->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        const Node* t2 = terminator;
        switch (entry.structured.tag) {
            case NotAStructured_construct: error("")
            case Structured_construct_If_TAG: {
                terminator = if_instr(a, entry.structured.payload.if_instr);
                break;
            }
            case Structured_construct_Match_TAG: {
                terminator = match_instr(a, entry.structured.payload.match_instr);
                break;
            }
            case Structured_construct_Loop_TAG: {
                terminator = loop_instr(a, entry.structured.payload.loop_instr);
                break;
            }
            case Structured_construct_Control_TAG: {
                terminator = control(a, entry.structured.payload.control);
                break;
            }
        }
        set_abstraction_body((Node*) get_structured_construct_tail(terminator), t2);
    }
    return terminator;
}

const Node* finish_body(BodyBuilder* bb, const Node* terminator) {
    assert(bb->mem && !bb->block_entry_mem);
    terminator = build_body(bb, terminator);
    destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* finish_body_with_return(BodyBuilder* bb, Nodes args) {
    return finish_body(bb, fn_ret(bb->arena, (Return) {
        .args = args,
        .mem = bb_mem(bb)
    }));
}

const Node* finish_body_with_unreachable(BodyBuilder* bb) {
    return finish_body(bb, unreachable(bb->arena, (Unreachable) {
        .mem = bb_mem(bb)
    }));
}

const Node* finish_body_with_selection_merge(BodyBuilder* bb, Nodes args) {
    return finish_body(bb, merge_selection(bb->arena, (MergeSelection) {
        .args = args,
        .mem = bb_mem(bb),
    }));
}

const Node* finish_body_with_loop_continue(BodyBuilder* bb, Nodes args)  {
    return finish_body(bb, merge_continue(bb->arena, (MergeContinue) {
        .args = args,
        .mem = bb_mem(bb),
    }));
}

const Node* finish_body_with_loop_break(BodyBuilder* bb, Nodes args) {
    return finish_body(bb, merge_break(bb->arena, (MergeBreak) {
        .args = args,
        .mem = bb_mem(bb),
    }));
}

const Node* finish_body_with_join(BodyBuilder* bb, const Node* jp, Nodes args) {
    return finish_body(bb, join(bb->arena, (Join) {
        .join_point = jp,
        .args = args,
        .mem = bb_mem(bb),
    }));
}

const Node* finish_body_with_jump(BodyBuilder* bb, const Node* target, Nodes args) {
    return finish_body(bb, jump(bb->arena, (Jump) {
        .target = target,
        .args = args,
        .mem = bb_mem(bb),
    }));
}

const Node* yield_value_and_wrap_in_block(BodyBuilder* bb, const Node* value) {
    IrArena* a = bb->arena;
    if (!bb->tail_block && entries_count_list(bb->stack) == 0) {
        const Node* last_mem = bb_mem(bb);
        cancel_body(bb);
        if (last_mem)
            return mem_and_value(a, (MemAndValue) {
                .mem = last_mem,
                .value = value
            });
        return value;
    }
    assert(bb->block_entry_mem && "This builder wasn't started with 'begin_block'");
    bb->tail_block->payload.basic_block.insert = bb;
    const Node* r = mem_and_value(bb->arena, (MemAndValue) {
        .mem = bb_mem(bb),
        .value = value
    });
    return r;
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    return yield_value_and_wrap_in_block(bb, maybe_tuple_helper(bb->arena, values));
}

const Node* finish_block_body(BodyBuilder* bb, const Node* terminator) {
    assert(bb->block_entry_mem);
    terminator = build_body(bb, terminator);
    destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder* bb, const Node* instruction) {
    size_t stack_size = entries_count_list(bb->stack);
    if (stack_size == 0) {
        cancel_body(bb);
        return instruction;
    }
    bind_internal(bb, instruction, 0);
    return yield_value_and_wrap_in_block(bb, instruction);
}

const Node* yield_values_and_wrap_in_compound_instruction(BodyBuilder* bb, Nodes values) {
    IrArena* arena = bb->arena;
    assert(!bb->mem && !bb->block_entry_mem && entries_count_list(bb->stack) == 0);
    cancel_body(bb);
    return maybe_tuple_helper(arena, values);
}

static Nodes gen_variables(BodyBuilder* bb, Nodes yield_types) {
    IrArena* a = bb->arena;

    Nodes qyield_types = add_qualifiers(a, yield_types, false);
    LARRAY(const Node*, tail_params, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++)
        tail_params[i] = param(a, qyield_types.nodes[i], NULL);
    return nodes(a, yield_types.count, tail_params);
}

Nodes add_structured_construct(BodyBuilder* bb, Nodes params, Structured_constructTag tag, union NodesUnion payload) {
    Node* tail = basic_block(bb->arena, params, NULL);
    StackEntry entry = {
        .structured = {
            .tag = tag,
            .payload = payload,
        },
        .vars = params,
    };
    switch (entry.structured.tag) {
        case NotAStructured_construct: error("")
        case Structured_construct_If_TAG: {
            entry.structured.payload.if_instr.tail = tail;
            entry.structured.payload.if_instr.mem = bb_mem(bb);
            break;
        }
        case Structured_construct_Match_TAG: {
            entry.structured.payload.match_instr.tail = tail;
            entry.structured.payload.match_instr.mem = bb_mem(bb);
            break;
        }
        case Structured_construct_Loop_TAG: {
            entry.structured.payload.loop_instr.tail = tail;
            entry.structured.payload.loop_instr.mem = bb_mem(bb);
            break;
        }
        case Structured_construct_Control_TAG: {
            entry.structured.payload.control.tail = tail;
            entry.structured.payload.control.mem = bb_mem(bb);
            break;
        }
    }
    bb->mem = get_abstraction_mem(tail);
    append_list(StackEntry , bb->stack, entry);
    bb->tail_block = tail;
    return entry.vars;
}

static Nodes gen_structured_construct(BodyBuilder* bb, Nodes yield_types, Structured_constructTag tag, union NodesUnion payload) {
    return add_structured_construct(bb, gen_variables(bb, yield_types), tag, payload);
}

Nodes gen_if(BodyBuilder* bb, Nodes yield_types, const Node* condition, const Node* true_case, Node* false_case) {
    return gen_structured_construct(bb, yield_types, Structured_construct_If_TAG, (union NodesUnion) {
        .if_instr = {
            .condition = condition,
            .if_true = true_case,
            .if_false = false_case,
            .yield_types = yield_types,
        }
    });
}

Nodes gen_match(BodyBuilder* bb, Nodes yield_types, const Node* inspectee, Nodes literals, Nodes cases, Node* default_case) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Match_TAG, (union NodesUnion) {
        .match_instr = {
            .yield_types = yield_types,
            .inspect = inspectee,
            .literals = literals,
            .cases = cases,
            .default_case = default_case
        }
    });
}

Nodes gen_loop(BodyBuilder* bb, Nodes yield_types, Nodes initial_args, Node* body) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Loop_TAG, (union NodesUnion) {
        .loop_instr = {
            .yield_types = yield_types,
            .initial_args = initial_args,
            .body = body
        },
    });
}

Nodes gen_control(BodyBuilder* bb, Nodes yield_types, Node* body) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Control_TAG, (union NodesUnion) {
        .control = {
            .yield_types = yield_types,
            .inside = body
        },
    });
}

begin_control_t begin_control(BodyBuilder* bb, Nodes yield_types) {
    IrArena* a = bb->arena;
    const Type* jp_type = qualified_type(a, (QualifiedType) {
            .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
            .is_uniform = true
    });
    const Node* jp = param(a, jp_type, NULL);
    Node* c = case_(a, singleton(jp));
    return (begin_control_t) {
        .results = gen_control(bb, yield_types, c),
        .case_ = c,
        .jp = jp
    };
}

begin_loop_helper_t begin_loop_helper(BodyBuilder* bb, Nodes yield_types, Nodes arg_types, Nodes initial_values) {
    assert(arg_types.count == initial_values.count);
    IrArena* a = bb->arena;
    begin_control_t outer_control = begin_control(bb, yield_types);
    BodyBuilder* outer_control_case_builder = begin_body_with_mem(a, get_abstraction_mem(outer_control.case_));
    LARRAY(const Node*, params, arg_types.count);
    for (size_t i = 0; i < arg_types.count; i++) {
        params[i] = param(a, qualified_type_helper(arg_types.nodes[i], false), NULL);
    }
    Node* loop_header = case_(a, nodes(a, arg_types.count, params));
    set_abstraction_body(outer_control.case_, finish_body_with_jump(outer_control_case_builder, loop_header, initial_values));
    BodyBuilder* loop_header_builder = begin_body_with_mem(a, get_abstraction_mem(loop_header));
    begin_control_t inner_control = begin_control(loop_header_builder, arg_types);
    set_abstraction_body(loop_header, finish_body_with_jump(loop_header_builder, loop_header, inner_control.results));

    return (begin_loop_helper_t) {
        .results = outer_control.results,
        .params = nodes(a, arg_types.count, params),
        .loop_body = inner_control.case_,
        .break_jp = outer_control.jp,
        .continue_jp = inner_control.jp,
    };
}

void cancel_body(BodyBuilder* bb) {
    for (size_t i = 0; i < entries_count_list(bb->stack); i++) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        // if (entry.structured.tag != NotAStructured_construct)
        //     destroy_list(entry.structured.stack);
    }
    destroy_list(bb->stack);
    //destroy_list(bb->stack_stack);
    free(bb);
}
