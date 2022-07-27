#include "ir_gen_helpers.h"

#include "list.h"

#include "../arena.h"
#include "../portability.h"
#include "../type.h"
#include "../block_builder.h"

Nodes gen_primop(BlockBuilder* instructions, PrimOp prim_op_) {
    const Node* instruction = prim_op(instructions->arena, prim_op_);
    Nodes output_types = unwrap_multiple_yield_types(instructions->arena, instruction->type);

    LARRAY(const char*, names, output_types.count);
    for (size_t i = 0; i < output_types.count; i++)
        names[i] = format_string(instructions->arena, "%s_out", primop_names[prim_op_.op]);


    if (output_types.count > 0)
        instruction = let(instructions->arena,  instruction, output_types.count, names);
    append_block(instructions, instruction);

    return output_types.count > 0 ? instruction->payload.let.variables : nodes(instructions->arena, 0, NULL);
}

void gen_push_value_stack(BlockBuilder* instructions, const Node* value) {
    append_block(instructions, prim_op(instructions->arena, (PrimOp) {
        .op = push_stack_op,
        .operands = nodes(instructions->arena, 2, (const Node*[]) { without_qualifier(value->type), value })
    }));
}

void gen_push_values_stack(BlockBuilder* instructions, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(instructions, value);
    }
}

const Node* gen_pop_value_stack(BlockBuilder* instructions, String var_name, const Type* type) {
    const char* names[] = { var_name };
    const Node* let_i = let(instructions->arena, prim_op(instructions->arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(instructions->arena, 1, (const Node*[]) { type })
    }), 1, names);
    append_block(instructions, let_i);
    return let_i->payload.let.variables.nodes[0];
}

Nodes gen_pop_values_stack(BlockBuilder* instructions, String var_name, const Nodes types) {
    LARRAY(const Node*, tmp, types.count);
    for (size_t i = 0; i < types.count; i++) {
        tmp[i] = gen_pop_value_stack(instructions, format_string(instructions->arena, "%s_%d", var_name, (int) i), types.nodes[i]);
    }
    return nodes(instructions->arena, types.count, tmp);
}

const Node* gen_load(BlockBuilder* instructions, const Node* ptr) {
    return gen_primop(instructions, (PrimOp) {
        .op = load_op,
        .operands = nodes(instructions->arena, 1, (const Node* []) { ptr })
    }).nodes[0];
}

void gen_store(BlockBuilder* instructions, const Node* ptr, const Node* value) {
    gen_primop(instructions, (PrimOp) {
        .op = store_op,
        .operands = nodes(instructions->arena, 2, (const Node* []) { ptr, value })
    });
}

const Node* gen_lea(BlockBuilder* instructions, const Node* base, const Node* offset, Nodes selectors) {
    LARRAY(const Node*, ops, 2 + selectors.count);
    ops[0] = base;
    ops[1] = offset;
    for (size_t i = 0; i < selectors.count; i++)
        ops[2 + i] = selectors.nodes[i];
    return gen_primop(instructions, (PrimOp) {
        .op = lea_op,
        .operands = nodes(instructions->arena, 2 + selectors.count, ops)
    }).nodes[0];
}
