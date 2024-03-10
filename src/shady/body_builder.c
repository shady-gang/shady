#include "ir_private.h"
#include "log.h"
#include "portability.h"
#include "type.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

BodyBuilder* begin_body(IrArena* a) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .instructions_list = new_list(const Node*),
    };
    return bb;
}

static void destroy_bb(BodyBuilder* bb) {
    destroy_list(bb->instructions_list);
    free(bb);
}

static Nodes create_output_variables(IrArena* a, Nodes types, String const output_names[]) {
    LARRAY(Node*, vars, types.count);
    for (size_t i = 0; i < types.count; i++) {
        String var_name = output_names ? output_names[i] : NULL;
        vars[i] = (Node*) var(a, types.nodes[i], var_name);
    }
    return nodes(a, types.count, (const Node**) vars);
}

static Nodes extract_results(BodyBuilder* bb, const Node* value, size_t outputs_count) {
    IrArena* a = bb->arena;

    if (outputs_count == SIZE_MAX) {
        assert(bb->arena->config.check_types && value->type);
        outputs_count = unwrap_multiple_yield_types(a, value->type).count;
    }

    switch (outputs_count) {
        case 0: return empty(a);
        case 1: return singleton(value);
        default: {
            LARRAY(const Node*, results, outputs_count);
            for (size_t i = 0; i < outputs_count; i++) {
                const Node* extract_instr = prim_op_helper(a, extract_op, empty(a), mk_nodes(a, value, int32_literal(a, i)));
                append_list(const Node*, bb->instructions_list, extract_instr);
                results[i] = extract_instr;
            }
            return nodes(a, outputs_count, results);
        }
    }
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, size_t outputs_count, String const output_names[]) {
    if (bb->arena->config.check_types) {
        assert(is_instruction(instruction));
    }
    Nodes params = extract_results(bb, instruction, outputs_count);
    append_list(const Node*, bb->instructions_list, instruction);
    return params;
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, SIZE_MAX, NULL);
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    assert(output_names);
    return bind_internal(bb, instruction, SIZE_MAX, output_names);
}

Nodes bind_instruction_explicit_result_types(BodyBuilder* bb, const Node* instruction, Nodes provided_types, String const output_names[]) {
    return bind_internal(bb, instruction, provided_types.count, output_names);
}

Nodes create_mutable_variables(BodyBuilder* bb, const Node* instruction, Nodes provided_types, String const output_names[]) {
    Nodes mutable_vars = create_output_variables(bb->arena, provided_types, output_names);
    const Node* let_mut_instr = let_mut(bb->arena, instruction, mutable_vars);
    return bind_internal(bb, let_mut_instr, 0, NULL);
}

Nodes bind_instruction_outputs_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count, String const output_names[]) {
    return bind_internal(bb, instruction, outputs_count, output_names);
}

static Nodes collect_instructions(BodyBuilder* bb) {
    size_t count = entries_count_list(bb->instructions_list);
    Nodes instructions = nodes(bb->arena, count, read_list(const Node*, bb->instructions_list));
    clear_list(bb->instructions_list);
    return instructions;
}

const Node* finish_body(BodyBuilder* bb, const Node* terminator) {
    IrArena* a = bb->arena;
    Nodes instructions = collect_instructions(bb);
    const Node* b = body(a, (Body) {
        .instructions = instructions,
        .terminator = terminator
    });

    destroy_bb(bb);
    return b;
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    IrArena* a = bb->arena;
    Nodes types = get_values_types(a, values);
    return region(a, (Region) {
        .yield_types = types,
        .body = finish_body(bb, region_end(a, (RegionEnd) {
            .args = values,
        }))
    });
}

const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, const Node* instruction, const Nodes* types) {
    size_t stack_size = entries_count_list(bb->instructions_list);
    if (stack_size == 0) {
        cancel_body(bb);
        return instruction;
    }
    Nodes bound = bind_internal(bb, instruction, types ? types->count : SIZE_MAX, NULL);
    return yield_values_and_wrap_in_block(bb, bound);
}

const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder* bb, const Node* instruction) {
    return bind_last_instruction_and_wrap_in_block_explicit_return_types(bb, instruction, NULL);
}

void cancel_body(BodyBuilder* bb) {
    destroy_list(bb->instructions_list);
    free(bb);
}

Nodes create_structured_if(BodyBuilder* bb, Nodes yield_types, const Node* condition, const Node* true_case, const Node* false_case) {
    IrArena* a = bb->arena;
    const Node* instruction = structured_if(a, (If) {
        .yield_types = yield_types,
        .condition = condition,
        .if_true = true_case,
        .if_false = false_case
    });
    return bind_instruction(bb, instruction);
}

Nodes create_structured_match(BodyBuilder* bb, Nodes yield_types, const Node* inspect, Nodes literals, Nodes cases, const Node* default_case) {
    IrArena* a = bb->arena;
    const Node* instruction = structured_match(a, (Match) {
        .yield_types = yield_types,
        .inspect = inspect,
        .literals = literals,
        .cases = cases,
        .default_case = default_case,
    });
    return bind_instruction(bb, instruction);
}

Nodes create_structured_loop(BodyBuilder* bb, Nodes yield_types, Nodes initial_values, const Node* iter_case) {
    IrArena* a = bb->arena;
    const Node* instruction = structured_loop(a, (Loop) {
        .yield_types = yield_types,
        .initial_args = initial_values,
        .body = iter_case,
    });
    return bind_instruction(bb, instruction);
}

