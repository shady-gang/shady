#ifndef SHADY_FREE_VARIABLES_H
#define SHADY_FREE_VARIABLES_H

#include "shady/ir.h"

typedef struct Scope_ Scope;
typedef struct CFNode_ CFNode;

typedef struct {
    CFNode* node;
    struct Dict* bound_set;
    struct Dict* free_set;
} CFNodeVariables;

struct Dict* compute_scope_variables_map(const Scope* scope);
void destroy_scope_variables_map(struct Dict*);

#endif
