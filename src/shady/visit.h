#ifndef SHADY_VISIT_H
#define SHADY_VISIT_H

#include "shady/ir.h"

typedef struct Visitor_ Visitor;
typedef void (*VisitNodeFn)(Visitor*, const Node*);
typedef void (*VisitOpFn)(Visitor*, NodeClass, const Node*);

struct Visitor_ {
   VisitNodeFn visit_fn;
   VisitOpFn visit_op;
};

void visit_node_operands(Visitor*, NodeClass exclude, const Node*);
void visit_module(Visitor* visitor, Module*);

void visit_node(Visitor* visitor, const Node*);
void visit_nodes(Visitor* visitor, Nodes nodes);

void visit_op(Visitor* visitor, NodeClass, const Node*);
void visit_ops(Visitor* visitor, NodeClass, Nodes nodes);

// visits the abstractions in the function, _except_ for the entry block (ie the function itself)
void visit_function_rpo(Visitor* visitor, const Node* function);
// use this in visit_node_operands to avoid visiting nodes in non-rpo order
#define IGNORE_ABSTRACTIONS_MASK NcBasic_block | NcAnon_lambda | NcDeclaration

#endif
