#include "ir_gen_helpers.h"

#include "list.h"

#include "../arena.h"
#include "../local_array.h"
#include "../type.h"

#include <assert.h>

Instructions begin_instructions(IrArena* arena) {
    return (Instructions) {
        .arena = arena,
        .list = new_list(const Node*)
    };
}

Nodes finish_instructions(Instructions i) {
    Nodes n = list_to_nodes(i.arena, i.list);
    destroy_list(i.list);
    return n;
}

void append_instr(Instructions i, const Node* l) {
    assert(l->tag == Let_TAG);
    append_list(const Node*, i.list, l);
}

void copy_instructions(Instructions in, Nodes l) {
    for (size_t i = 0; i < l.count; i++)
        append_instr(in, l.nodes[i]);
}

const Node* wrap_in_let(IrArena* arena, const Node* node) {
    return let(arena, (Let) {
        .instruction = node,
        .variables = nodes(arena, 0, NULL),
    });
}

void gen_push_value_stack(Instructions instructions, const Node* value) {
    append_instr(instructions, let(instructions.arena, (Let) {
        .variables = nodes(instructions.arena, 0, NULL),
        .instruction = prim_op(instructions.arena, (PrimOp) {
            .op = push_stack_op,
            .operands = nodes(instructions.arena, 2, (const Node*[]) { without_qualifier(value->type), value })
        })
    }));
}

void gen_push_values_stack(Instructions instructions, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(instructions, value);
    }
}

void gen_push_fn_stack(Instructions instructions, const Node* fn_ptr) {
    const Type* ret_param_type = int_type(instructions.arena);

    append_instr(instructions, let(instructions.arena, (Let) {
        .variables = nodes(instructions.arena, 0, NULL),
        .instruction = prim_op(instructions.arena, (PrimOp) {
            .op = push_stack_uniform_op,
            .operands = nodes(instructions.arena, 2, (const Node*[]) { ret_param_type, fn_ptr })
        })
    }));
}

const Node* gen_pop_fn_stack(Instructions instructions, String var_name) {
    const Type* ret_param_type = int_type(instructions.arena);
    const Type* q_ret_param_type = qualified_type(instructions.arena, (QualifiedType) {.type = ret_param_type, .is_uniform = true});

    const Node* ret_tmp_vars[] = { var(instructions.arena, q_ret_param_type, var_name)};
    append_instr(instructions, let(instructions.arena, (Let) {
        .variables = nodes(instructions.arena, 1, ret_tmp_vars),
        .instruction = prim_op(instructions.arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(instructions.arena, 1, (const Node*[]) { ret_param_type })
        })
    }));
    return ret_tmp_vars[0];
}

const Node* gen_pop_value_stack(Instructions instructions, String var_name, const Type* type) {
    const Type* q_type = qualified_type(instructions.arena, (QualifiedType) {.type = type, .is_uniform = false});
    const Node* ret_tmp_vars[] = { var(instructions.arena, q_type, var_name)};
    append_instr(instructions, let(instructions.arena, (Let) {
        .variables = nodes(instructions.arena, 1, ret_tmp_vars),
        .instruction = prim_op(instructions.arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(instructions.arena, 1, (const Node*[]) { type })
        })
    }));
    return ret_tmp_vars[0];
}

Nodes gen_pop_values_stack(Instructions instructions, String var_name, const Nodes types) {
    LARRAY(const Node*, tmp, types.count);
    for (size_t i = 0; i < types.count; i++) {
        tmp[i] = gen_pop_value_stack(instructions, format_string(instructions.arena, "%s_%d", var_name, (int) i), types.nodes[i]);
    }
    return nodes(instructions.arena, types.count, tmp);
}