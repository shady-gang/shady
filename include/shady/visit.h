#ifndef SHADY_VISIT_H
#define SHADY_VISIT_H

#include "shady/ir/grammar.h"

typedef struct Visitor_ Visitor;
typedef void (*VisitNodeFn)(Visitor*, const Node*);
typedef void (*VisitOpFn)(Visitor*, NodeClass, String, const Node*, size_t);

struct Visitor_ {
    VisitNodeFn visit_node_fn;
    VisitOpFn visit_op_fn;
};

void shd_visit_node_operands(Visitor* visitor, NodeClass exclude, const Node* node);
void shd_visit_module(Visitor* visitor, Module* mod);

void shd_visit_node(Visitor* visitor, const Node* node);
void shd_visit_nodes(Visitor* visitor, Nodes nodes);

void shd_visit_op(Visitor* visitor, NodeClass op_class, String op_name, const Node* op, size_t i);
void shd_visit_ops(Visitor* visitor, NodeClass op_class, String op_name, Nodes ops);

// visits the abstractions in the function, starting with the entry block (ie the function itself)
void shd_visit_function_rpo(Visitor* visitor, const Node* function);
void shd_visit_function_bodies_rpo(Visitor* visitor, const Node* function);

#endif
