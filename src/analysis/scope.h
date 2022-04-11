#ifndef SHADY_SCOPE_H

#include "ir.h"

#include "list.h"
#include "dict.h"

typedef struct Scope_ {
    const Node* entry;
    struct List* contents;
    struct Dict* succs;
    struct Dict* preds;
} Scope;

struct List* build_scopes(const Node* root);

void dispose_scope(Scope*);

#define SHADY_SCOPE_H

#endif
