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
LTNode* looptree_lookup(LoopTree*, const Node* block);

void destroy_loop_tree(LoopTree* lt);

LoopTree* build_loop_tree(CFG* s);
void dump_loop_trees(FILE* output, Module* mod);

#endif // SHADY_LOOPTREE_H
