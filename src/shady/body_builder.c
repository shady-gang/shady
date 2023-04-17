#include "ir_private.h"
#include "log.h"
#include "portability.h"
#include "type.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

typedef struct {
    const Node* instr;
    Nodes vars;
    bool mut;
} StackEntry;

BodyBuilder* begin_body(IrArena* a) {
    BodyBuilder* bb = malloc(sizeof(BodyBuilder));
    *bb = (BodyBuilder) {
        .arena = a,
        .stack = new_list(StackEntry),
    };
    return bb;
}

static Nodes create_output_variables(IrArena* a, const Node* value, size_t outputs_count, Nodes* provided_types, String const output_names[]) {
    Nodes types;
    if (a->config.check_types) {
        types = unwrap_multiple_yield_types(a, value->type);
        if (provided_types) {
            assert(provided_types->count == types.count);
            // Check that the types we got are subtypes of what we care about
            for (size_t i = 0; i < types.count; i++)
                assert(is_subtype(provided_types->nodes[i], types.nodes[i]));
            types = *provided_types;
        }
        outputs_count = types.count;
    } else {
        assert(outputs_count != SIZE_MAX);
        if (provided_types) {
            assert(provided_types->count == outputs_count);
            types = *provided_types;
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
        if (!var_name) {
            if (value->tag == PrimOp_TAG) {
                var_name = primop_names[value->payload.prim_op.op];
            } else {
                var_name = node_tags[value->tag];
            }
        }
        vars[i] = (Node*) var(a, types.nodes[i], var_name);
    }

    // for (size_t i = 0; i < outputs_count; i++) {
    //     vars[i]->payload.var.instruction = value;
    //     vars[i]->payload.var.output = i;
    // }
    return nodes(a, outputs_count, (const Node**) vars);
}

static Nodes bind_internal(BodyBuilder* bb, const Node* instruction, bool mut, size_t outputs_count, Nodes* provided_types, String const output_names[]) {
    if (bb->arena->config.check_types) {
        assert(is_instruction(instruction));
    }
    Nodes params = create_output_variables(bb->arena, instruction, outputs_count, provided_types, output_names);
    StackEntry entry = {
        .instr = instruction,
        .vars = params,
        .mut = mut,
    };
    append_list(StackEntry, bb->stack, entry);
    return params;
}

Nodes bind_instruction_extra(BodyBuilder* bb, const Node* instruction, size_t outputs_count, Nodes* provided_types, String const output_names[]) {
    return bind_internal(bb, instruction, false, outputs_count, provided_types, output_names);
}

Nodes bind_instruction_extra_mutable(BodyBuilder* bb, const Node* instruction, size_t outputs_count, Nodes* provided_types, String const output_names[]) {
    return bind_internal(bb, instruction, true, outputs_count, provided_types, output_names);
}

Nodes bind_instruction_named(BodyBuilder* bb, const Node* instruction, String const output_names[]) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, false, SIZE_MAX, NULL, output_names);
}

Nodes bind_instruction(BodyBuilder* bb, const Node* instruction) {
    assert(bb->arena->config.check_types);
    return bind_internal(bb, instruction, false, SIZE_MAX, NULL, NULL);
}

void bind_variables(BodyBuilder* bb, Nodes vars, Nodes values) {
    StackEntry entry = {
        .instr = quote_helper(bb->arena, values),
        .vars = vars,
        .mut = false,
    };
    append_list(StackEntry, bb->stack, entry);
}

const Node* finish_body(BodyBuilder* bb, const Node* terminator) {
    size_t stack_size = entries_count_list(bb->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = read_list(StackEntry, bb->stack)[i];
        const Node* lam = lambda(bb->arena, entry.vars, terminator);
        terminator = (entry.mut ? let_mut : let)(bb->arena, entry.instr, lam);
    }

    destroy_list(bb->stack);
    free(bb);
    return terminator;
}

const Node* yield_values_and_wrap_in_block(BodyBuilder* bb, Nodes values) {
    IrArena* arena = bb->arena;
    const Node* terminator = yield(arena, (Yield) { .args = values });
    const Node* lam = lambda(arena, empty(arena), finish_body(bb, terminator));
    return block(arena, (Block) {
        .yield_types = get_values_types(arena, values),
        .inside = lam,
    });
}

void cancel_body(BodyBuilder* bb) {
    destroy_list(bb->stack);
    free(bb);
}
