#ifndef SHADY_SCOPE_H

#include "ir.h"

#include "list.h"

typedef struct Scope_ {
    const Node* entry;
    struct List* contents;
} Scope;

struct List* build_scopes(const Node* root);

#define SHADY_SCOPE_H

#endif
