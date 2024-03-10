#ifndef SHADY_SCOPE_H

#include "shady/ir.h"

typedef struct CFNode_ CFNode;

typedef enum {
    JumpEdge,
    LetTailEdge,
    StructuredEnterBodyEdge,
    StructuredLeaveBodyEdge,
    /// Join points might leak, and as a consequence, there might be no static edge to the
    /// tail of the enclosing let, which would make it look like dead code.
    /// This edge type accounts for that risk, they can be ignored where more precise info is available
    /// (see is_control_static for example)
    StructuredPseudoExitEdge,
} CFEdgeType;

typedef struct {
    CFEdgeType type;
    CFNode* src;
    CFNode* dst;
} CFEdge;

typedef enum {
    CFNodeType_EntryNode,
    CFNodeType_BBNode,
    CFNodeType_CaseNode,
    CFNodeType_Tail,
} CFNodeType;

struct CFNode_ {
    CFNodeType type;
    const Node* abstraction;
    const Node* body;
    /// Specifies which part of the body (which instructions) this CFNode accounts for
    struct {
        size_t start, end;
    } range;

    CFNode* parent;
    CFNode* tail;
    // const Node* node;

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
    struct Dict* structurally_dominates;
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
    struct Dict* abs_map;

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

CFNode* scope_lookup(Scope*, const Node* abs);
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
