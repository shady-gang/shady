#include "ir_private.h"
#include "rewrite.h"
#include "fold.h"
#include "log.h"
#include "portability.h"

#include "../type.h"

#include "list.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

typedef struct {
    const Node* instr;
    Node* tail;
    bool mut;
} StackEntry;

BodyBuilder* begin_body(IrArena* arena) {
    BodyBuilder* builder = malloc(sizeof(BodyBuilder));
    *builder = (BodyBuilder) {
        .arena = arena,
        .stack = new_list(StackEntry),
    };
    return builder;
}

static Nodes create_output_variables(IrArena* arena, const Node* value, size_t outputs_count, Nodes* provided_types, const char* output_names[]) {
    Nodes types;
    if (arena->config.check_types) {
        types = unwrap_multiple_yield_types(arena, value->type);
        if (provided_types) {
            assert(provided_types->count == types.count);
            // Check that the types we got are subtypes of what we care about
            for (size_t i = 0; i < types.count; i++)
                assert(is_subtype(provided_types->nodes[i], types.nodes[i]));
            types = *provided_types;
        }
        outputs_count = types.count;
    } else {
         if (provided_types) {
            assert(provided_types->count == outputs_count);
            types = *provided_types;
        } else {
            LARRAY(const Type*, nulls, outputs_count);
            for (size_t i = 0; i < outputs_count; i++)
                nulls[i] = NULL;
            types = nodes(arena, outputs_count, nulls);
        }
    }

    LARRAY(Node*, vars, types.count);
    for (size_t i = 0; i < types.count; i++)
        vars[i] = (Node*) var(arena, types.nodes[i], output_names ? output_names[i] : node_tags[value->tag]);

    for (size_t i = 0; i < outputs_count; i++) {
        vars[i]->payload.var.instruction = value;
        vars[i]->payload.var.output = i;
    }
    return nodes(arena, outputs_count, (const Node**) vars);
}

static Nodes bind_internal(BodyBuilder* builder, const Node* instruction, bool mut, size_t outputs_count, Nodes* provided_types, const char* output_names[]) {
    if (is_value(instruction))
        instruction = quote(builder->arena, instruction);
    Nodes params = create_output_variables(builder->arena, instruction, outputs_count, provided_types, output_names);
    StackEntry entry = {
        .instr = instruction,
        .tail = lambda(builder->arena, params),
        .mut = mut,
    };
    append_list(StackEntry, builder->stack, entry);
    return params;
}

Nodes bind_instruction_extra(BodyBuilder* builder, const Node* instruction, size_t outputs_count, Nodes* provided_types, const char* output_names[]) {
    return bind_internal(builder, instruction, false, outputs_count, provided_types, output_names);
}

Nodes bind_instruction_extra_mutable(BodyBuilder* builder, const Node* instruction, size_t outputs_count, Nodes* provided_types, const char* output_names[]) {
    return bind_internal(builder, instruction, true, outputs_count, provided_types, output_names);
}

Nodes bind_instruction(BodyBuilder* builder, const Node* instruction) {
    assert(builder->arena->config.check_types);
    return bind_internal(builder, instruction, false, SIZE_MAX, NULL, NULL);
}

#undef arena

const Node* finish_body(BodyBuilder* builder, const Node* terminator) {
    size_t stack_size = entries_count_list(builder->stack);
    for (size_t i = stack_size - 1; i < stack_size; i--) {
        StackEntry entry = read_list(StackEntry, builder->stack)[i];
        entry.tail->payload.lam.body = terminator;
        terminator = let(builder->arena, entry.mut, entry.instr, entry.tail);
    }

    destroy_list(builder->stack);
    free(builder);
    return terminator;
}
