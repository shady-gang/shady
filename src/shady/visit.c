#include "shady/ir.h"
#include "shady/visit.h"
#include "analysis/cfg.h"

#include "log.h"
#include "portability.h"

#include <assert.h>

void shd_visit_node(Visitor* visitor, const Node* node) {
    assert(visitor->visit_node_fn);
    if (node)
        visitor->visit_node_fn(visitor, node);
}

void shd_visit_nodes(Visitor* visitor, Nodes nodes) {
    for (size_t i = 0; i < nodes.count; i++) {
        shd_visit_node(visitor, nodes.nodes[i]);
    }
}

void shd_visit_op(Visitor* visitor, NodeClass op_class, String op_name, const Node* op, size_t i) {
    if (!op)
        return;
    if (visitor->visit_op_fn)
        visitor->visit_op_fn(visitor, op_class, op_name, op, i);
    else
        visitor->visit_node_fn(visitor, op);
}

void shd_visit_ops(Visitor* visitor, NodeClass op_class, String op_name, Nodes ops) {
    for (size_t i = 0; i < ops.count; i++)
        shd_visit_op(visitor, op_class, op_name, ops.nodes[i], i);
}

void shd_visit_function_rpo(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    CFG* cfg = build_fn_cfg(function);
    assert(cfg->rpo[0]->node == function);
    for (size_t i = 0; i < cfg->size; i++) {
        const Node* node = cfg->rpo[i]->node;
        shd_visit_node(visitor, node);
    }
    shd_destroy_cfg(cfg);
}

void shd_visit_function_bodies_rpo(Visitor* visitor, const Node* function) {
    assert(function->tag == Function_TAG);
    CFG* cfg = build_fn_cfg(function);
    assert(cfg->rpo[0]->node == function);
    for (size_t i = 0; i < cfg->size; i++) {
        const Node* node = cfg->rpo[i]->node;
        assert(is_abstraction(node));
        if (get_abstraction_body(node))
            shd_visit_node(visitor, get_abstraction_body(node));
    }
    shd_destroy_cfg(cfg);
}

#pragma GCC diagnostic error "-Wswitch"

void shd_visit_module(Visitor* visitor, Module* mod) {
    Nodes decls = shd_module_get_all_exported(mod);
    shd_visit_nodes(visitor, decls);
}

void shd_visit_node_operands_generated(Visitor* visitor, NodeClass exclude, const Node* node);

void shd_visit_node_operands(Visitor* visitor, NodeClass exclude, const Node* node) {
    shd_visit_node_operands_generated(visitor, exclude, node);
    for (size_t i = 0; i < node->annotations.count; i++) {
        if ((exclude & NcAnnotation) == 0)
            shd_visit_op(visitor, NcAnnotation, "annotation", node->annotations.nodes[i], i);
    }
}

#include "visit_generated.c"
