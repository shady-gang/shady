#ifndef SHADY_CFG_H
#define SHADY_CFG_H

#include "shady/ir.h"

typedef struct CFNode_ CFNode;

typedef enum {
    JumpEdge,
    StructuredEnterBodyEdge,
    StructuredLeaveBodyEdge,
    /// Join points might leak, and as a consequence, there might be no static edge to the
    /// tail of the enclosing let, which would make it look like dead code.
    /// This edge type accounts for that risk, they can be ignored where more precise info is available
    /// (see is_control_static for example)
    StructuredTailEdge,
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
    struct Dict* structurally_dominates;
};

typedef struct Arena_ Arena;
typedef struct CFG_ {
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
} CFG;

/**
 * @returns @ref List of @ref CFG*
 */
struct List* build_cfgs(Module*);

typedef struct LoopTree_ LoopTree;

/** Construct the CFG starting in node.
 */
CFG* build_cfg(const Node* fn, const Node* entry, LoopTree* lt, bool flipped);

/** Construct the CFG starting in node.
 * Dominance will only be computed with respect to the nodes reachable by @p entry.
 */
#define build_fn_cfg(node) build_cfg(node, node, NULL, false)

/** Construct the CFG stating in Node.
 * Dominance will only be computed with respect to the nodes reachable by @p entry.
 * This CFG will contain post dominance information instead of regular dominance!
 */
#define build_fn_cfg_flipped(node) build_cfg(node, node, NULL, true)

CFNode* cfg_lookup(CFG* cfg, const Node* abs);
void compute_rpo(CFG*);
void compute_domtree(CFG*);

bool is_cfnode_structural_target(CFNode*);

CFNode* least_common_ancestor(CFNode* i, CFNode* j);

void destroy_cfg(CFG* cfg);

#endif
