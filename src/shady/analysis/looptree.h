#ifndef SHADY_LOOPTREE_H
#define SHADY_LOOPTREE_H

#include "cfg.h"

// Loop tree implementation based on Thorin, translated to C and somewhat simplified
// https://github.com/AnyDSL/thorin

typedef struct LTNode_ LTNode;

struct LTNode_ {
    enum { LF_HEAD, LF_LEAF } type;
    LTNode* parent;

    /**
     * @ref List of @ref CFNode*
     */
    struct List* cf_nodes;

    int depth;

    /**
     * @ref List of @ref LTNode*
     */
    struct List* lf_children;
};

typedef struct LoopTree_ LoopTree;

struct LoopTree_ {
    LTNode* root;

    /**
     * @ref Dict from const @ref Node* to @ref LTNode*
     */
    struct Dict* map;
};

/**
 * Returns the leaf for this node.
 */
LTNode* shd_loop_tree_lookup(LoopTree* lt, const Node* block);

void shd_destroy_loop_tree(LoopTree* lt);

LoopTree* shd_new_loop_tree(CFG* s);

#endif // SHADY_LOOPTREE_H
