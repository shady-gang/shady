#ifndef SHADY_LEAK_H
#define SHADY_LEAK_H

#include <stdbool.h>

#include "uses.h"
#include "cfg.h"

typedef void (VisitEnclosingAbsCallback)(void*, const Use*);
void visit_enclosing_abstractions(UsesMap*, const Node*, void* uptr, VisitEnclosingAbsCallback fn);

bool is_control_static(const UsesMap*, const Node* control);
/// Returns the Control node that defines the join point, or NULL if it's defined by something else
const Node* get_control_for_jp(const UsesMap*, const Node* jp);

#endif
