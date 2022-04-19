#ifndef SHADY_SCOPE_H

#include "shady/ir.h"

#include "list.h"
#include "dict.h"

typedef struct CFNode_ CFNode;
struct CFNode_ {
    const Node* node;
    struct List* succs;
    struct List* preds;
    // set by compute_rpo
    size_t rpo_index;
    // set by compute_domtree
    CFNode* idom;
    struct List* dominates;
};

typedef struct Scope_ {
    size_t size;
    struct List* contents;
    CFNode* entry;
    // set by compute_rpo
    CFNode** rpo;
} Scope;

struct List* build_scopes(const Node* root);
Scope build_scope(const Node* entry);

void compute_rpo(Scope* scope);
void compute_domtree(Scope* scope);

void dispose_scope(Scope*);

#define SHADY_SCOPE_H

#endif
