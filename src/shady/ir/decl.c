#include "shady/ir/decl.h"
#include "shady/rewrite.h"

#include "../ir_private.h"

#include <assert.h>

bool shd_compare_nodes(Nodes* a, Nodes* b);

Node* shd_constant(Module* mod, Constant payload) {
    IrArena* arena = mod->arena;
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Constant_TAG,
        .payload.constant = payload
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}

Node* shd_global_var(Module* mod, GlobalVariable payload) {
    IrArena* arena = mod->arena;
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = GlobalVariable_TAG,
        .payload.global_variable = payload
    };
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    _shd_module_add_decl(mod, decl);
    return decl;
}

const Node* shd_find_or_process_decl(Rewriter* rewriter, const char* name) {
    Nodes old_decls = shd_module_get_declarations(rewriter->src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (strcmp(get_declaration_name(decl), name) == 0) {
            return shd_rewrite_node(rewriter, decl);
        }
    }
    assert(false);
}
