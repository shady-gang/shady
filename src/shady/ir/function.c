#include "shady/ir/function.h"

#include "../ir_private.h"

#include <assert.h>

Node* param(IrArena* arena, const Type* type, const char* name) {
    Param param = {
        .type = type,
        .name = string(arena, name),
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

Node* function(Module* mod, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    assert(!mod->sealed);
    IrArena* arena = mod->arena;
    Function payload = {
        .module = mod,
        .params = params,
        .body = NULL,
        .name = name,
        .annotations = annotations,
        .return_types = return_types,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Function_TAG,
        .payload.fun = payload
    };
    Node* fn = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, fn);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = fn;
        param->payload.param.pindex = i;
    }

    return fn;
}

Node* basic_block(IrArena* arena, Nodes params, const char* name) {
    BasicBlock payload = {
        .params = params,
        .body = NULL,
        .name = name,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = BasicBlock_TAG,
        .payload.basic_block = payload
    };

    Node* bb = _shd_create_node_helper(arena, node, NULL);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = bb;
        param->payload.param.pindex = i;
    }

    return bb;
}