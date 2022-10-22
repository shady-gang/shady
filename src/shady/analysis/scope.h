#ifndef SHADY_SCOPE_H

#include "shady/ir.h"

typedef struct CFNode_ CFNode;

typedef enum {
    ForwardEdge,
    LetTailEdge,
    ControlBodyEdge,
    IfBodyEdge,
    MatchBodyEdge,
    LoopBodyEdge,
} CFEdgeType;

typedef struct {
    CFEdgeType type;
    CFNode* src;
    CFNode* dst;
} CFEdge;

struct CFNode_ {
    const Node* node;
    /// Edges where this node is the source
    struct List* succ_edges;
    /// Edges where this node is the destination
    struct List* pred_edges;
    // set by compute_rpo
    size_t rpo_index;
    // set by compute_domtree
    CFNode* idom;
    struct List* dominates;
};

typedef struct Arena_ Arena;
typedef struct Scope_ {
    Arena* arena;
    size_t size;
    struct List* contents;
    CFNode* entry;
    // set by compute_rpo
    CFNode** rpo;
} Scope;

struct List* build_scopes(const Node* root);
Scope build_scope(const Node*);

void compute_rpo(Scope* scope);
void compute_domtree(Scope* scope);

void dispose_scope(Scope*);

#define SHADY_SCOPE_H

#endif
