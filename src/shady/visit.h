#ifndef SHADY_VISIT_H
#define SHADY_VISIT_H

#include "shady/ir.h"

typedef struct Visitor_ Visitor;
typedef void (*VisitFn)(Visitor*, const Node*);

struct Visitor_ {
   VisitFn visit_fn;
   // Enabling this will make visit_children build the scope of functions and look at their continuations in RPO
   bool visit_fn_scope_rpo;
   // Enabling this will make visit_children visit control flow targets
   bool visit_cf_targets;
   // Enabling this will make visit_children visit references to other declarations (visit_module will still visit those at the top)
   bool visit_referenced_decls;
};

void visit_children(Visitor*, const Node*);
void visit_fn_blocks_except_head(Visitor*, const Node*);
void visit_nodes(Visitor* visitor, Nodes nodes);
void visit_module(Visitor* visitor, Module*);

#endif
