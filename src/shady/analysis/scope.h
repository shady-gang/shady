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
    BlockBodyEdge,
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
    struct Dict* structurally_dominated;
};

typedef struct Arena_ Arena;
typedef struct Scope_ {
    Arena* arena;
    size_t size;
    struct List* contents;
    struct Dict* map;
    CFNode* entry;
    // set by compute_rpo
    CFNode** rpo;
} Scope;

struct List* build_scopes(Module*);
Scope* new_scope(const Node*);

CFNode* scope_lookup(Scope*, const Node* block);
void compute_rpo(Scope*);
void compute_domtree(Scope*);

void destroy_scope(Scope*);

#define SHADY_SCOPE_H

#endif
