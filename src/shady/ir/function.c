#include "shady/ir/function.h"

#include "../ir_private.h"

#include <assert.h>

Node* _shd_param(IrArena* arena, const Type* type) {
    Param param = {
        .type = type,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Param_TAG,
        .payload.param = param
    };
    return _shd_create_node_helper(arena, node, NULL);
}

Node* shd_function(Module* mod, Function payload) {
    IrArena* arena = mod->arena;
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Function_TAG,
        .payload.fun = payload
    };
    Node* fn = _shd_create_node_helper(arena, node, NULL);

    for (size_t i = 0; i < payload.params.count; i++) {
        Node* param = (Node*) payload.params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = fn;
        param->payload.param.pindex = i;
    }

    return fn;
}

Node* shd_basic_block(IrArena* arena, BasicBlock payload) {
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = BasicBlock_TAG,
        .payload.basic_block = payload
    };

    Node* bb = _shd_create_node_helper(arena, node, NULL);

    for (size_t i = 0; i < payload.params.count; i++) {
        Node* param = (Node*) payload.params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = bb;
        param->payload.param.pindex = i;
    }

    return bb;
}

const Node* shd_get_abstraction_mem(const Node* abs) {
    return abs_mem(abs->arena, (AbsMem) { .abs = abs });
}

String shd_get_abstraction_name(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        default: assert(false);
    }
}

String shd_get_abstraction_name_unsafe(const Node* abs) {
    assert(is_abstraction(abs));
    switch (abs->tag) {
        case Function_TAG: return abs->payload.fun.name;
        case BasicBlock_TAG: return abs->payload.basic_block.name;
        default: assert(false);
    }
}

String shd_get_abstraction_name_safe(const Node* abs) {
    String name = shd_get_abstraction_name_unsafe(abs);
    if (name)
        return name;
    return shd_fmt_string_irarena(abs->arena, "%%%d", abs->id);
}

void shd_set_abstraction_body(Node* abs, const Node* body) {
    assert(is_abstraction(abs));
    assert(!body || is_terminator(body));
    IrArena* a = abs->arena;

    if (body) {
        while (true) {
            const Node* mem0 = shd_get_original_mem(get_terminator_mem(body));
            assert(mem0->tag == AbsMem_TAG);
            Node* mem_abs = mem0->payload.abs_mem.abs;
            if (is_basic_block(mem_abs)) {
                BodyBuilder* insert = mem_abs->payload.basic_block.insert;
                if (insert && mem_abs != abs) {
                    const Node* mem = _shd_bb_insert_mem(insert);
                    const Node* block = _shd_bb_insert_block(insert);
                    shd_set_abstraction_body((Node*) block, _shd_bld_finish_pseudo_instr(insert, body));
                    body = jump_helper(a, mem, block, shd_empty(a));
                    // mem_abs->payload.basic_block.insert = NULL;
                    continue;
                }
            }
            assert(mem_abs == abs);
            break;
        }
    }

    switch (abs->tag) {
        case Function_TAG: abs->payload.fun.body = body; break;
        case BasicBlock_TAG: abs->payload.basic_block.body = body; break;
        default: assert(false);
    }
}

Nodes shd_bld_call(BodyBuilder* bb, const Node* callee, Nodes args) {
    assert(shd_get_arena_config(shd_get_bb_arena(bb))->check_types);
    const Node* instruction = indirect_call(shd_get_bb_arena(bb), (IndirectCall) { .callee = callee, .args = args, .mem = shd_bld_mem(bb) });
    return shd_bld_add_instruction_extract(bb, instruction);
}
