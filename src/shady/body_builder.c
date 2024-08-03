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

BodyBuilder* begin_body(IrArena* a) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .stack = new_list(StackEntry),
    };
    return bb;
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, size_t outputs_count) {
    if (bb->arena->config.check_types) {
        assert(is_instruction(instruction) || is_value(instruction));
    }
    if (is_instruction(instruction)) {
        StackEntry entry = {
            .vars = empty(bb->arena),
            .structured.payload.let = {
                .instruction = instruction,
            }
        };
        append_list(StackEntry, bb->stack, entry);
    }
    if (outputs_count > 1) {
        LARRAY(const Node*, extracted, outputs_count);
        for (size_t i = 0; i < outputs_count; i++)
            extracted[i] = gen_extract_single(bb, instruction, int32_literal(bb->arena, i));
        return nodes(bb->arena, outputs_count, extracted);
    } else if (outputs_count == 1)
        return singleton(instruction);
    else
        return empty(bb->arena);
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, unwrap_multiple_yield_types(bb->arena, instruction->type).count);
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    assert(output_names);
    return bind_internal(bb, instruction, unwrap_multiple_yield_types(bb->arena, instruction->type).count);
}

const Node* bind_identifiers(IrArena* arena, const Node* instruction, bool mut, Strings names, Nodes types);

Nodes parser_create_mutable_variables(BodyBuilder* bb, const Node* instruction, Nodes provided_types, Strings output_names) {
    const Node* let_mut_instr = bind_identifiers(bb->arena, instruction, true, output_names, provided_types);
    return bind_internal(bb, let_mut_instr, 0);
}

Nodes parser_create_immutable_variables(BodyBuilder* bb, const Node* instruction, Strings output_names) {
    const Node* let_mut_instr = bind_identifiers(bb->arena, instruction, false, output_names, empty(bb->arena));
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
            case NotAStructured_construct:
                terminator = let(a, entry.structured.payload.let.instruction, terminator);
                break;
            case Structured_construct_If_TAG: {
                Node* tail = case_(bb->arena, entry.vars);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.if_instr.tail = tail;
                terminator = if_instr(a, entry.structured.payload.if_instr);
                break;
            }
            case Structured_construct_Match_TAG: {
                Node* tail = case_(bb->arena, entry.vars);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.match_instr.tail = tail;
                terminator = match_instr(a, entry.structured.payload.match_instr);
                break;
            }
            case Structured_construct_Loop_TAG: {
                Node* tail = case_(bb->arena, entry.vars);
                set_abstraction_body(tail, terminator);
                entry.structured.payload.loop_instr.tail = tail;
                terminator = loop_instr(a, entry.structured.payload.loop_instr);
                break;
            }
            case Structured_construct_Control_TAG: {
                Node* tail = case_(bb->arena, entry.vars);
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
    terminator = build_body(bb, terminator);
    destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* yield_values_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, Nodes values, const Nodes types) {
    IrArena* arena = bb->arena;
    const Node* terminator = block_yield(arena, (BlockYield) { .args = values });
    const Node* block_case = case_(arena, empty(arena));
    set_abstraction_body(block_case, finish_body(bb, terminator));
    return block(arena, (Block) {
        .yield_types = types,
        .inside = block_case,
    });
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    return yield_values_and_wrap_in_block_explicit_return_types(bb, values, get_values_types(bb->arena, values));
}

const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, const Node* instruction, const Nodes types) {
    size_t stack_size = entries_count_list(bb->stack);
    if (stack_size == 0) {
        cancel_body(bb);
        return instruction;
    }
    Nodes bound = bind_internal(bb, instruction, types.count);
    return yield_values_and_wrap_in_block_explicit_return_types(bb, bound, types);
}

const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder* bb, const Node* instruction) {
    return bind_last_instruction_and_wrap_in_block_explicit_return_types(bb, instruction, unwrap_multiple_yield_types(bb->arena, instruction->type));
}

static Nodes finish_with_instruction_list(BodyBuilder* bb) {
    IrArena* a = bb->arena;
    size_t count = entries_count_list(bb->stack);
    LARRAY(const Node*, list, count);
    for (size_t i = 0; i < count; i++) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        if (entry.structured.tag != NotAStructured_construct) {
            error("When using a BodyBuilder to create compound instructions, control flow is not allowed.")
        }
        list[i] = entry.structured.payload.let.instruction;
    }

    destroy_list(bb->stack);
    free(bb);
    return nodes(a, count, list);
}

const Node* yield_values_and_wrap_in_compound_instruction_explicit_return_types(BodyBuilder* bb, Nodes values, const Nodes types) {
    IrArena* arena = bb->arena;
    return compound_instruction(arena, finish_with_instruction_list(bb), values);
}

const Node* yield_values_and_wrap_in_compound_instruction(BodyBuilder* bb, Nodes values) {
    return yield_values_and_wrap_in_compound_instruction_explicit_return_types(bb, values, get_values_types(bb->arena, values));
}

const Node* bind_last_instruction_and_wrap_in_compound_instruction_explicit_return_types(BodyBuilder* bb, const Node* instruction, const Nodes types) {
    IrArena* arena = bb->arena;
    Nodes values = bind_instruction_outputs_count(bb, instruction, types.count);
    return compound_instruction(arena, finish_with_instruction_list(bb), values);
}

const Node* bind_last_instruction_and_wrap_in_compound_instruction(BodyBuilder* bb, const Node* instruction) {
    return bind_last_instruction_and_wrap_in_compound_instruction_explicit_return_types(bb, instruction, unwrap_multiple_yield_types(bb->arena, instruction->type));
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
