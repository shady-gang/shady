#ifndef SHADY_VISIT_H
#define SHADY_VISIT_H

#include "shady/ir.h"

typedef struct Visitor_ Visitor;
typedef void (*VisitFn)(Visitor*, const Node*);

struct Visitor_ {
   VisitFn visit_fn;
   // Enabling this will make visit_children build the scope of functions and look at their continuations in RPO
   bool visit_fn_scope_rpo;
   // Enabling this will make visit_children visit continuation targets (inside terminators), be wary this could cause infinite loops
   bool visit_continuations;

   bool visit_return_fn_annotation;
   bool visit_fn_addr;
   bool visit_referenced_decls;
};

void visit_children(Visitor*, const Node*);
void visit_fn_blocks_except_head(Visitor*, const Node*);

#endif
