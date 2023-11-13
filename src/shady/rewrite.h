#ifndef SHADY_REWRITE_H
#define SHADY_REWRITE_H

#include "shady/ir.h"

typedef struct Rewriter_ Rewriter;

typedef const Node* (*RewriteNodeFn)(Rewriter*, const Node*);
typedef const Node* (*RewriteOpFn)(Rewriter*, NodeClass, String, const Node*);

const Node* rewrite_node(Rewriter*, const Node*);
const Node* rewrite_node_with_fn(Rewriter*, const Node*, RewriteNodeFn);

const Node* rewrite_op(Rewriter*, NodeClass, String, const Node*);
const Node* rewrite_op_with_fn(Rewriter*, NodeClass, String, const Node*, RewriteOpFn);

/// Applies the rewriter to all nodes in the collection
Nodes rewrite_nodes(Rewriter*, Nodes);
Nodes rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteNodeFn fn);

Nodes rewrite_ops(Rewriter*, NodeClass, String, Nodes);
Nodes rewrite_ops_with_fn(Rewriter* rewriter, NodeClass,String, Nodes values, RewriteOpFn fn);

Strings import_strings(IrArena*, Strings);

struct Rewriter_ {
    RewriteNodeFn rewrite_fn;
    RewriteOpFn rewrite_op_fn;
    IrArena* src_arena;
    IrArena* dst_arena;
    Module* src_module;
    Module* dst_module;
    struct {
        bool search_map;
        bool write_map;
        bool rebind_let;
        bool fold_quote;
        bool process_variables;
    } config;
    struct Dict* map;
    struct Dict* decls_map;
};

Rewriter create_rewriter(Module* src, Module* dst, RewriteNodeFn fn);
Rewriter create_importer(Module* src, Module* dst);
Rewriter create_substituter(Module* arena);
void destroy_rewriter(Rewriter*);

void rewrite_module(Rewriter*);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* recreate_node_identity(Rewriter*, const Node*);

/// Rewrites a constant / function header
Node* recreate_decl_header_identity(Rewriter*, const Node*);
void  recreate_decl_body_identity(Rewriter*, const Node*, Node*);

/// Rewrites a variable under a new identity
const Node* recreate_variable(Rewriter* rewriter, const Node* old);
Nodes recreate_variables(Rewriter* rewriter, Nodes old);

/// Looks up if the node was already processed
const Node* search_processed(const Rewriter*, const Node*);
/// Same as search_processed but asserts if it fails to find a mapping
const Node* find_processed(const Rewriter*, const Node*);
void register_processed(Rewriter*, const Node*, const Node*);
void register_processed_list(Rewriter*, Nodes, Nodes);
void clear_processed_non_decls(Rewriter*);

#endif
