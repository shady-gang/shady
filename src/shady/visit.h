#ifndef SHADY_VISIT_H
#define SHADY_VISIT_H

#include "shady/ir.h"

typedef struct Visitor_ Visitor;
typedef void (*VisitFn)(Visitor*, const Node*);

struct Visitor_ {
   VisitFn visit_fn;
   // Enabling this will make visit_children build the scope of functions and look at their continuations in RPO
   bool visit_fn_scope_rpo;
   // Enabling this will make visit_children visit targets of control flow terminators, be wary this could cause infinite loops
   bool visit_cf_targets;
   bool visit_return_fn_annotation;
};

void visit_children(Visitor*, const Node*);
void visit_fn_blocks_except_head(Visitor*, const Node*);

#endif
