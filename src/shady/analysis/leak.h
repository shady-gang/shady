#ifndef SHADY_LEAK_H
#define SHADY_LEAK_H

#include "cfg.h"

#include "shady/analysis/uses.h"

#include <stdbool.h>

typedef void (VisitEnclosingAbsCallback)(void*, const Use*);
void shd_visit_enclosing_abstractions(UsesMap* map, const Node* n, void* uptr, VisitEnclosingAbsCallback fn);

bool shd_is_control_static(const UsesMap* map, const Node* control);
/// Returns the Control node that defines the join point, or NULL if it's defined by something else
const Node* shd_get_control_for_jp(const UsesMap* map, const Node* jp);

#endif
