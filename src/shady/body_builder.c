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
        .mem0 = mem,
        .mem = mem,
    };
    return bb;
}

BodyBuilder* begin_block_with_side_effects(IrArena* a) {
    Node* block = basic_block(a, empty(a), NULL);
    BodyBuilder* builder = begin_body_with_mem(a, get_abstraction_mem(block));
    builder->bb = block;
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
        assert(is_instruction(instruction) || is_value(instruction));
    }
    if (is_mem(instruction))
        bb->mem = instruction;
    return deconstruct_composite(bb->arena, bb, instruction, outputs_count);
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, singleton(instruction->type).count);
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    assert(output_names);
    return bind_internal(bb, instruction, singleton(instruction->type).count);
}

const Node* bind_identifiers(IrArena* arena, const Node* instruction, const Node* mem, bool mut, Strings names, Nodes types);

Nodes parser_create_mutable_variables(BodyBuilder* bb, const Node* instruction, Nodes provided_types, Strings output_names) {
    const Node* let_mut_instr = bind_identifiers(bb->arena, instruction, bb->mem, true, output_names, provided_types);
    return bind_internal(bb, let_mut_instr, 0);
}

Nodes parser_create_immutable_variables(BodyBuilder* bb, const Node* instruction, Strings output_names) {
    const Node* let_mut_instr = bind_identifiers(bb->arena, instruction, bb->mem, false, output_names, empty(bb->arena));
    return bind_internal(bb, let_mut_instr, 0);
}

Nodes bind_instruction_outputs_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count) {
    return bind_internal(bb, instruction, outputs_count);
}

static const Node* build_body(BodyBuilder* bb, const Node* terminator) {
    IrArena* a = bb->arena;
    size_t stack_size = entries_count_list(bb->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        switch (entry.structured.tag) {
            case NotAStructured_construct: error("")
            case Structured_construct_If_TAG: {
                Node* tail = basic_block(bb->arena, entry.vars, NULL);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.if_instr.tail = tail;
                terminator = if_instr(a, entry.structured.payload.if_instr);
                break;
            }
            case Structured_construct_Match_TAG: {
                Node* tail = basic_block(bb->arena, entry.vars, NULL);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.match_instr.tail = tail;
                terminator = match_instr(a, entry.structured.payload.match_instr);
                break;
            }
            case Structured_construct_Loop_TAG: {
                Node* tail = basic_block(bb->arena, entry.vars, NULL);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.loop_instr.tail = tail;
                terminator = loop_instr(a, entry.structured.payload.loop_instr);
                break;
            }
            case Structured_construct_Control_TAG: {
                Node* tail = basic_block(bb->arena, entry.vars, NULL);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.control.tail = tail;
                terminator = control(a, entry.structured.payload.control);
                break;
            }
        }
    }
    return terminator;
}

const Node* finish_body(BodyBuilder* bb, const Node* terminator) {
    assert(bb->mem0);
    terminator = build_body(bb, terminator);
    destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* yield_value_and_wrap_in_block(BodyBuilder* bb, const Node* value) {
    IrArena* a = bb->arena;
    if (entries_count_list(bb->stack) == 0) {
        const Node* last_mem = bb_mem(bb);
        cancel_body(bb);
        if (last_mem)
            return mem_and_value(a, (MemAndValue) {
                .mem = last_mem,
                .value = value
            });
        return value;
    }
    assert(bb->bb && "This builder wasn't started with 'begin_block'");
    bb->bb->payload.basic_block.insert = bb;
    const Node* r = mem_and_value(bb->arena, (MemAndValue) {
        .mem = bb_mem(bb),
        .value = value
    });
    return r;
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    return yield_value_and_wrap_in_block(bb, maybe_tuple_helper(bb->arena, values));
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
    assert(!bb->mem0 && !bb->stack);
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
    StackEntry entry = {
        .structured = {
            .tag = tag,
            .payload = payload,
        },
        .vars = params,
    };
    append_list(StackEntry , bb->stack, entry);
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
