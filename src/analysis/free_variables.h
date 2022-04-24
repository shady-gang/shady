#ifndef SHADY_FREE_VARIABLES_H
#define SHADY_FREE_VARIABLES_H

#include "scope.h"
#include "list.h"

struct List* compute_free_variables(const Node*);

#endif
