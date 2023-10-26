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

    /** @brief Edges where this node is the source
     *
     * @ref List of @ref CFEdge
     */
    struct List* succ_edges;

    /** @brief Edges where this node is the destination
     *
     * @ref List of @ref CFEdge
     */
    struct List* pred_edges;

    // set by compute_rpo
    size_t rpo_index;

    // set by compute_domtree
    CFNode* idom;

    /** @brief All Nodes directly dominated by this CFNode.
     *
     * @ref List of @ref CFNode*
     */
    struct List* dominates;
    struct Dict* structurally_dominated;
};

typedef struct Arena_ Arena;
typedef struct Scope_ {
    Arena* arena;
    size_t size;
    bool flipped;

    /**
     * @ref List of @ref CFNode*
     */
    struct List* contents;

    /**
     * @ref Dict from const @ref Node* to @ref CFNode*
     */
    struct Dict* map;

    CFNode* entry;
    // set by compute_rpo
    CFNode** rpo;
} Scope;

/**
 * @returns @ref List of @ref Scope*
 */
struct List* build_scopes(Module*);

typedef struct LoopTree_ LoopTree;

/** Construct the scope stating in Node.
 */
Scope* new_scope_impl(const Node* entry, LoopTree* lt, bool flipped);

#define new_scope_lt(node, lt) new_scope_impl(node, lt, false);
#define new_scope_lt_flipped(node, lt) new_scope_impl(node, lt, true);

Scope* new_scope_lt_impl(const Node* entry, LoopTree* lt, bool flipped);

/** Construct the scope starting in Node.
 * Dominance will only be computed with respect to the nodes reachable by @p entry.
 */
#define new_scope(node) new_scope_impl(node, NULL, false);

/** Construct the scope stating in Node.
 * Dominance will only be computed with respect to the nodes reachable by @p entry.
 * This scope will contain post dominance information instead of regular dominance!
 */
#define new_scope_flipped(node) new_scope_impl(node, NULL, true);

CFNode* scope_lookup(Scope*, const Node* block);
void compute_rpo(Scope*);
void compute_domtree(Scope*);

CFNode* least_common_ancestor(CFNode* i, CFNode* j);

void destroy_scope(Scope*);

/**
 * @returns @ref List of @ref CFNode*
 */
struct List* scope_get_dom_frontier(Scope*, const CFNode* node);

#define SHADY_SCOPE_H

#endif
