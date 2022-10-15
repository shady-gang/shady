#ifndef SHADY_SCOPE_H

#include "shady/ir.h"

typedef struct CFNode_ CFNode;

typedef enum {
    UncoloredEdge,
    ForwardEdge,
    CallcReturnEdge,
    SelectionMergeTargetEdge,
    LoopBreakEdge,
    LoopContinueEdge,
    ControlJoinTargetEdge,
    BackEdge,
} CFEdgeType;

typedef struct {
    CFEdgeType type;
    CFNode* src;
    CFNode* dst;
} CFEdge;

typedef struct {
    /// The head (function node)
    const Node* head;
    /// The body we are referring to
    /// (a given block might have multiple bodies associated with it in the case of structured CF)
    const Node* body;
    size_t offset;
} CFLocation;

struct CFNode_ {
    CFLocation location;
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
Scope build_scope(CFLocation entry_location);
Scope build_scope_from_basic_block(const Node*);

void compute_rpo(Scope* scope);
void compute_domtree(Scope* scope);

void dispose_scope(Scope*);

#define SHADY_SCOPE_H

#endif
