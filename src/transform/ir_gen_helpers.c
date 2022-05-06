#include "ir_gen_helpers.h"

#include "list.h"

#include "../arena.h"
#include "../portability.h"
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

Nodes gen_primop(Instructions instructions, PrimOp prim_op_) {
    Nodes output_types = typecheck_primop(instructions.arena, prim_op_);

    LARRAY(const Node*, outputs, output_types.count);
    for (size_t i = 0; i < output_types.count; i++)
        outputs[i] = var(instructions.arena, output_types.nodes[i], format_string(instructions.arena, "%s_out", primop_names[prim_op_.op]));

    append_instr(instructions, let(instructions.arena, (Let) {
        .variables = nodes(instructions.arena, output_types.count, outputs),
        .instruction = prim_op(instructions.arena, prim_op_)
    }));

    return nodes(instructions.arena, output_types.count, outputs);
}

const Node* gen_load(Instructions instructions, const Node* ptr) {
    return gen_primop(instructions, (PrimOp) {
        .op = load_op,
        .operands = nodes(instructions.arena, 1, (const Node* []) { ptr })
    }).nodes[0];
}

void gen_store(Instructions instructions, const Node* ptr, const Node* value) {
    gen_primop(instructions, (PrimOp) {
        .op = store_op,
        .operands = nodes(instructions.arena, 2, (const Node* []) { ptr, value })
    });
}

const Node* gen_lea(Instructions instructions, const Node* base, const Node* offset, Nodes selectors) {
    LARRAY(const Node*, ops, 2 + selectors.count);
    ops[0] = base;
    ops[1] = offset;
    for (size_t i = 0; i < selectors.count; i++)
        ops[2 + i] = selectors.nodes[i];
    return gen_primop(instructions, (PrimOp) {
        .op = lea_op,
        .operands = nodes(instructions.arena, 2 + selectors.count, ops)
    }).nodes[0];
}
