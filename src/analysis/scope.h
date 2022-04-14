#ifndef SHADY_SCOPE_H

#include "ir.h"

#include "list.h"
#include "dict.h"

typedef struct CFNode_ {
    const Node* node;
    struct List* succs;
    struct List* preds;
    size_t rpo_index;
} CFNode;

typedef struct Scope_ {
    size_t size;
    struct List* contents;
    const CFNode* entry;
    const CFNode** rpo;
} Scope;

struct List* build_scopes(const Node* root);

void dispose_scope(Scope*);

#define SHADY_SCOPE_H

#endif
