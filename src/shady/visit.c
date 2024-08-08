#include "shady/ir.h"
#include "log.h"
#include "visit.h"
#include "analysis/cfg.h"

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

void visit_op(Visitor* visitor, NodeClass op_class, String op_name, const Node* op) {
    if (!op)
        return;
    if (visitor->visit_op_fn)
        visitor->visit_op_fn(visitor, op_class, op_name, op);
    else
        visitor->visit_node_fn(visitor, op);
}

void visit_ops(Visitor* visitor, NodeClass op_class, String op_name, Nodes ops) {
    for (size_t i = 0; i < ops.count; i++)
        visit_op(visitor, op_class, op_name, ops.nodes[i]);
}

void visit_function_rpo(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    CFG* cfg = build_fn_cfg(function);
    assert(cfg->rpo[0]->node == function);
    for (size_t i = 1; i < cfg->size; i++) {
        const Node* node = cfg->rpo[i]->node;
        visit_node(visitor, node);
    }
    destroy_cfg(cfg);
}

#pragma GCC diagnostic error "-Wswitch"

#include "visit_generated.c"

void visit_module(Visitor* visitor, Module* mod) {
    Nodes decls = get_module_declarations(mod);
    visit_nodes(visitor, decls);
}
