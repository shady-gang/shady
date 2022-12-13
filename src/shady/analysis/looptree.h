#ifndef SHADY_LOOPTREE_H
#define SHADY_LOOPTREE_H

#include "scope.h"

// Loop tree implementation based on Thorin, translated to C and somewhat simplified
// https://github.com/AnyDSL/thorin

typedef struct LTNode_ LTNode;

struct LTNode_ {
    enum { LF_HEAD, LF_LEAF } type;
    LTNode* parent;
    struct List* cf_nodes;
    int depth;
    struct List* lf_children;
};

typedef struct {
    LTNode* root;
} LoopTree;

static void destroy_lt_node(LTNode* n);
void destroy_loop_tree(LoopTree* lt);

#endif // SHADY_LOOPTREE_H
