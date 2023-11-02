#include "shady/ir.h"
#include "log.h"
#include "visit.h"
#include "analysis/scope.h"

#include <assert.h>

void visit_node(Visitor* visitor, const Node* node) {
    assert(visitor->visit_node_fn);
    if (node)
        visitor->visit_node_fn(visitor, node);
}

void visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
         visit_node(visitor, nodes.nodes[i]);
    }
}

void visit_op(Visitor* visitor, NodeClass class, const Node* op) {
    if (!op)
        return;
    if (visitor->visit_op_fn)
        visitor->visit_op_fn(visitor, class, op);
    else
        visitor->visit_node_fn(visitor, op);
}

void visit_ops(Visitor* visitor, NodeClass class, Nodes ops) {
    for (size_t i = 0; i < ops.count; i++)
        visit_op(visitor, class, ops.nodes[i]);
}

void visit_function_rpo(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    Scope* scope = new_scope(function);
    assert(scope->rpo[0]->node == function);
    for (size_t i = 1; i < scope->size; i++) {
        const Node* node = scope->rpo[i]->node;
        visit_node(visitor, node);
    }
    destroy_scope(scope);
}

#pragma GCC diagnostic error "-Wswitch"

#include "visit_generated.c"

void visit_module(Visitor* visitor, Module* mod) {
    Nodes decls = get_module_declarations(mod);
    visit_nodes(visitor, decls);
}
