#ifndef SHADY_REWRITE_H
#define SHADY_REWRITE_H

#include "shady/ir.h"

typedef struct Rewriter_ Rewriter;

typedef const Node* (*RewriteFn)(Rewriter*, const Node*);
typedef void (*RewriteFnMut)(Rewriter*, const Node*, Node*);

/// Applies the rewriter to all nodes in the collection
Nodes rewrite_nodes(Rewriter*, Nodes);

/// bring in a node unmodified into a new arena
const Node* import_node   (IrArena*, const Node*);
Nodes       import_nodes  (IrArena*, Nodes);
Strings     import_strings(IrArena*, Strings);

struct Rewriter_ {
    IrArena* src_arena;
    IrArena* dst_arena;

    RewriteFn rewrite_fn;
    RewriteFnMut rewrite_decl_body;
    struct Dict* processed;
};

const Node* rewrite_node(Rewriter*, const Node*);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* recreate_node_identity(Rewriter*, const Node*);

Node* recreate_decl_header_identity(Rewriter*, const Node*);
void  recreate_decl_body_identity(Rewriter*, const Node*, Node*);

/// Looks up if the node was already processed
const Node* search_processed(const Rewriter*, const Node*);
/// Same as search_processed but asserts if it fails to find a mapping
const Node* find_processed(const Rewriter*, const Node*);
void register_processed(Rewriter*, const Node*, const Node*);

#endif
