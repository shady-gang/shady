#ifndef SHADY_CALLGRAPH
#define SHADY_CALLGRAPH

#include "shady/ir.h"

typedef struct CGNode_ CGNode;

typedef struct {
    CGNode* src_fn;
    CGNode* dst_fn;
    const Node* abs;
    const Node* instr;
} CGEdge;

struct CGNode_ {
    const Node* fn;
    struct Dict* callers;
    struct Dict* callees;
    struct {
        int index, lowlink;
        bool on_stack;
    } tarjan;

    bool is_recursive;
    /// set to true if the address of this is captured by a FnAddr node that is not immediately consumed by a call
    bool is_address_captured;
    bool calls_indirect;
};

typedef struct Callgraph_ {
    struct Dict* fn2cgn;
} CallGraph;

CallGraph* shd_new_callgraph(Module* mod);
void shd_destroy_callgraph(CallGraph* graph);

#endif
