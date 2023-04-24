#ifndef SHADY_FREE_VARIABLES_H
#define SHADY_FREE_VARIABLES_H

#include "shady/ir.h"

typedef struct Scope_ Scope;

struct List* compute_free_variables(const Scope* scope, const Node*);

#endif
