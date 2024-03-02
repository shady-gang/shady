#include "ir_private.h"
#include "log.h"
#include "portability.h"
#include "type.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

typedef struct {
    Structured_constructTag tag;
    union NodesUnion payload;
} StackEntry;

BodyBuilder* begin_body(IrArena* a) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .instructions_list = new_list(const Node*),
    };
    return bb;
}

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
        vars[i] = (Node*) var(a, types.nodes[i], var_name);
    }

    // for (size_t i = 0; i < outputs_count; i++) {
    //     vars[i]->payload.var.instruction = value;
    //     vars[i]->payload.var.output = i;
    // }
    return nodes(a, outputs_count, (const Node**) vars);
}

static Nodes extract_results(BodyBuilder* bb, const Node* value, size_t outputs_count, const Node** output_types) {
    IrArena* a = bb->arena;
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

    switch (types.count) {
        case 0: return empty(a);
        case 1: return singleton(value);
        default: {
            LARRAY(const Node*, results, types.count);
            for (size_t i = 0; i < types.count; i++) {
                const Node* extract_instr = prim_op_helper(a, extract_op, empty(a), mk_nodes(a, value, int32_literal(a, i)));
                append_list(const Node*, bb->instructions_list, extract_instr);
                results[i] = extract_instr;
            }
            return nodes(a, types.count, results);
        }
    }
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, size_t outputs_count, const Node** provided_types, String const output_names[]) {
    if (bb->arena->config.check_types) {
        assert(is_instruction(instruction));
    }
    Nodes params = extract_results(bb, instruction, outputs_count, provided_types);
    append_list(const Node*, bb->instructions_list, instruction);
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
    const Node* let_mut_instr = let_mut(bb->arena, instruction, mutable_vars);
    return bind_internal(bb, let_mut_instr, 0, NULL, NULL);
}

Nodes bind_instruction_outputs_count(BodyBuilder* bb, const Node* instruction, size_t outputs_count, String const output_names[]) {
    return bind_internal(bb, instruction, outputs_count, NULL, output_names);
}

static Nodes collect_instructions(BodyBuilder* bb) {
    size_t count = entries_count_list(bb->instructions_list);
    Nodes instructions = nodes(bb->arena, count, read_list(const Node*, bb->instructions_list));
    clear_list(bb->instructions_list);
    return instructions;
}

static void destroy_bb(BodyBuilder* bb) {
    destroy_list(bb->instructions_list);
    free(bb);
}

const Node* finish_body(BodyBuilder* bb, const Node* terminator) {
    Nodes instructions = collect_instructions(bb);
    const Node* b = body(bb->arena, (Body) {
        .instructions = instructions,
        .terminator = terminator
    });
    destroy_bb(bb);
    return b;
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    IrArena* a = bb->arena;
    bind_instruction(bb, quote_helper(a, values));

    Nodes instructions = collect_instructions(bb);
    const Node* i = compound_instruction(bb->arena, (CompoundInstruction) {
        .instructions = instructions
    });
    destroy_bb(bb);
    return i;
}

const Node* bind_last_instruction_and_wrap_in_block_explicit_return_types(BodyBuilder* bb, const Node* instruction, const Nodes* types) {
    size_t stack_size = entries_count_list(bb->instructions_list);
    if (stack_size == 0) {
        cancel_body(bb);
        return instruction;
    }
    Nodes bound = bind_internal(bb, instruction, types ? types->count : SIZE_MAX, types ? types->nodes : NULL, NULL);
    return yield_values_and_wrap_in_block(bb, bound);
}

const Node* bind_last_instruction_and_wrap_in_block(BodyBuilder* bb, const Node* instruction) {
    return bind_last_instruction_and_wrap_in_block_explicit_return_types(bb, instruction, NULL);
}

void cancel_body(BodyBuilder* bb) {
    destroy_list(bb->instructions_list);
    free(bb);
}

Nodes create_structured_if(BodyBuilder*, Nodes yield_types, const Node* condition, const Node* true_case, const Node* false_case);

Nodes create_structured_match(BodyBuilder*, Nodes yield_types, const Node* inspect, Nodes literals, Nodes cases, const Node* default_case);

Nodes create_structured_loop(BodyBuilder*, Nodes yield_types, Nodes initial_values, const Node* iter_case);

