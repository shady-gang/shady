#include "ir_private.h"
#include "log.h"
#include "portability.h"
#include "type.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

typedef struct {
    //struct List* stack;
    Structured_constructTag tag;
    union NodesUnion payload;
} BlockEntry;

typedef struct {
    const Node* instr;
    BlockEntry structured;
    Nodes vars;
} StackEntry;

BodyBuilder* begin_body(IrArena* a) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .stack = new_list(StackEntry),
        //.stack_stack = new_list(BlockEntry),
    };
    return bb;
}

const Node* var(IrArena* arena, const char* name, const Node* instruction, size_t i);

static Nodes create_output_variables(IrArena* a, const Node* value, size_t outputs_count, const Node** output_types, String const output_names[]) {
    Nodes types;
    if (a->config.check_types) {
        types = unwrap_multiple_yield_types(a, value->type);
        // outputs count has to match or not be given
        assert(outputs_count == types.count || outputs_count == SIZE_MAX);
        if (output_types) {
            // Check that the types we got are subtypes of what we care about
            for (size_t i = 0; i < types.count; i++)
                assert(is_subtype(output_types[i], types.nodes[i]));
            types = nodes(a, outputs_count, output_types);
        }
        outputs_count = types.count;
    } else {
        assert(outputs_count != SIZE_MAX);
        if (output_types) {
            types = nodes(a, outputs_count, output_types);
        } else {
            LARRAY(const Type*, nulls, outputs_count);
            for (size_t i = 0; i < outputs_count; i++)
                nulls[i] = NULL;
            types = nodes(a, outputs_count, nulls);
        }
    }

    LARRAY(Node*, vars, types.count);
    for (size_t i = 0; i < types.count; i++) {
        String var_name = output_names ? output_names[i] : NULL;
        vars[i] = (Node*) var(a, var_name, value, i);
    }

    // for (size_t i = 0; i < outputs_count; i++) {
    //     vars[i]->payload.var.instruction = value;
    //     vars[i]->payload.var.output = i;
    // }
    return nodes(a, outputs_count, (const Node**) vars);
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, size_t outputs_count, const Node** provided_types, String const output_names[]) {
    if (bb->arena->config.check_types) {
        assert(is_instruction(instruction));
    }
    Nodes params = create_output_variables(bb->arena, instruction, outputs_count, provided_types, output_names);
    StackEntry entry = {
        .instr = instruction,
        .vars = params,
    };
    append_list(StackEntry, bb->stack, entry);
    return params;
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, SIZE_MAX, NULL, NULL);
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    assert(output_names);
    return bind_internal(bb, instruction, SIZE_MAX, NULL, output_names);
}

Nodes bind_instruction_explicit_result_types(BodyBuilder* bb, const Node* instruction, Nodes provided_types, String const output_names[]) {
    return bind_internal(bb, instruction, provided_types.count, provided_types.nodes, output_names);
}

Nodes create_mutable_variables(BodyBuilder* bb, const Node* instruction, Nodes provided_types, String const output_names[]) {
    Nodes mutable_vars = create_output_variables(bb->arena, instruction, provided_types.count, provided_types.nodes, output_names);
    const Node* let_mut_instr = let_mut(bb->arena, instruction, mutable_vars, provided_types);
    return bind_internal(bb, let_mut_instr, 0, NULL, NULL);
}

Nodes bind_instruction_outputs_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count, String const output_names[]) {
    return bind_internal(bb, instruction, outputs_count, NULL, output_names);
}

void bind_variables(BodyBuilder* bb, Nodes vars, Nodes values) {
    StackEntry entry = {
        .instr = quote_helper(bb->arena, values),
        .vars = vars,
    };
    append_list(StackEntry, bb->stack, entry);
}

void bind_variables2(BodyBuilder* bb, Nodes vars, const Node* instr) {
    StackEntry entry = {
        .instr = instr,
        .vars = vars,
    };
    append_list(StackEntry, bb->stack, entry);
}

static const Node* build_body(BodyBuilder* bb, const Node* terminator) {
    IrArena* a = bb->arena;
    size_t stack_size = entries_count_list(bb->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        switch (entry.structured.tag) {
            case NotAStructured_construct:
                terminator = let(a, entry.instr, entry.vars, case_(bb->arena, empty(bb->arena), terminator));
                break;
            case Structured_construct_If_TAG:
                entry.structured.payload.if_instr.tail = case_(bb->arena, entry.vars, terminator);
                terminator = if_instr(a, entry.structured.payload.if_instr);
                break;
            case Structured_construct_Match_TAG:
                entry.structured.payload.match_instr.tail = case_(bb->arena, entry.vars, terminator);
                terminator = match_instr(a, entry.structured.payload.match_instr);
                break;
            case Structured_construct_Loop_TAG:
                entry.structured.payload.loop_instr.tail = case_(bb->arena, entry.vars, terminator);
                terminator = loop_instr(a, entry.structured.payload.loop_instr);
                break;
            case Structured_construct_Control_TAG:
                entry.structured.payload.control.tail = case_(bb->arena, entry.vars, terminator);
                terminator = control(a, entry.structured.payload.control);
                break;
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

const Node* yield_values_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, Nodes values, const Nodes* types) {
    IrArena* arena = bb->arena;
    assert(arena->config.check_types || types);
    const Node* terminator = block_yield(arena, (BlockYield) { .args = values });
    const Node* lam = case_(arena, empty(arena), finish_body(bb, terminator));
    return block(arena, (Block) {
        .yield_types = arena->config.check_types ? get_values_types(arena, values) : *types,
        .inside = lam,
    });
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    return yield_values_and_wrap_in_block_explicit_return_types(bb, values, NULL);
}

const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, const Node* instruction, const Nodes* types) {
    size_t stack_size = entries_count_list(bb->stack);
    if (stack_size == 0) {
        cancel_body(bb);
        return instruction;
    }
    Nodes bound = bind_internal(bb, instruction, types ? types->count : SIZE_MAX, types ? types->nodes : NULL, NULL);
    return yield_values_and_wrap_in_block_explicit_return_types(bb, bound, types);
}

const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder* bb, const Node* instruction) {
    return bind_last_instruction_and_wrap_in_block_explicit_return_types(bb, instruction, NULL);
}

static Nodes gen_structured_construct(BodyBuilder* bb, Nodes yield_types, Structured_constructTag tag, union NodesUnion payload) {
    IrArena* a = bb->arena;
    Nodes qyield_types = add_qualifiers(a, yield_types, false);
    LARRAY(const Node*, tail_params, yield_types.count);
    for (size_t i = 0; i < yield_types.count; i++)
        tail_params[i] = param(a, qyield_types.nodes[i], NULL);

    StackEntry entry = {
        .structured = {
            .tag = tag,
            .payload = payload,
        },
        .vars = nodes(a, yield_types.count, tail_params),
    };
    append_list(StackEntry , bb->stack, entry);
    return entry.vars;
}

Nodes gen_if(BodyBuilder* bb, Nodes yield_types, const Node* condition, const Node* true_case, const Node* false_case) {
    return gen_structured_construct(bb, yield_types, Structured_construct_If_TAG, (union NodesUnion) {
        .if_instr = {
            .condition = condition,
            .if_true = true_case,
            .if_false = false_case,
            .yield_types = yield_types,
        }
    });
}

Nodes gen_match(BodyBuilder* bb, Nodes yield_types, const Node* inspectee, Nodes literals, Nodes cases, const Node* default_case) {
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

Nodes gen_loop(BodyBuilder* bb, Nodes yield_types, Nodes initial_args, const Node* body) {
    return gen_structured_construct(bb, yield_types, Structured_construct_Loop_TAG, (union NodesUnion) {
        .loop_instr = {
            .yield_types = yield_types,
            .initial_args = initial_args,
            .body = body
        },
    });
}

Nodes gen_control(BodyBuilder* bb, Nodes yield_types, const Node* body) {
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
