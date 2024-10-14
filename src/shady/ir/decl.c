#include "shady/ir/decl.h"

#include "../ir_private.h"

#include <assert.h>

bool shd_compare_nodes(Nodes* a, Nodes* b);

Node* _shd_constant(Module* mod, Nodes annotations, const Type* hint, String name) {
    IrArena* arena = mod->arena;
    Constant cnst = {
        .annotations = annotations,
        .name = string(arena, name),
        .type_hint = hint,
        .value = NULL,
    };
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Constant_TAG,
        .payload.constant = cnst
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}

Node* _shd_global_var(Module* mod, Nodes annotations, const Type* type, const char* name, AddressSpace as) {
    const Node* existing = shd_module_get_declaration(mod, name);
    if (existing) {
        assert(existing->tag == GlobalVariable_TAG);
        assert(existing->payload.global_variable.type == type);
        assert(existing->payload.global_variable.address_space == as);
        assert(!mod->arena->config.check_types || shd_compare_nodes((Nodes*) &existing->payload.global_variable.annotations, &annotations));
        return (Node*) existing;
    }

    IrArena* arena = mod->arena;
    GlobalVariable gvar = {
        .annotations = annotations,
        .name = string(arena, name),
        .type = type,
        .address_space = as,
        .init = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = GlobalVariable_TAG,
        .payload.global_variable = gvar
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}