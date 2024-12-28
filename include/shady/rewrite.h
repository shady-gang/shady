#ifndef SHADY_REWRITE_H
#define SHADY_REWRITE_H

#include "shady/ir/grammar.h"

typedef struct Rewriter_ Rewriter;

typedef const Node* (*RewriteNodeFn)(Rewriter*, const Node*);
typedef const Node* (*RewriteOpFn)(Rewriter*, NodeClass, String, const Node*);

const Node* shd_rewrite_node(Rewriter* rewriter, const Node* node);
const Node* shd_rewrite_node_with_fn(Rewriter* rewriter, const Node* node, RewriteNodeFn fn);

const Node* shd_rewrite_op(Rewriter* rewriter, NodeClass class, String op_name, const Node* node);
const Node* shd_rewrite_op_with_fn(Rewriter* rewriter, NodeClass class, String op_name, const Node* node, RewriteOpFn fn);

/// Applies the rewriter to all nodes in the collection
Nodes shd_rewrite_nodes(Rewriter* rewriter, Nodes old_nodes);
Nodes shd_rewrite_nodes_with_fn(Rewriter* rewriter, Nodes values, RewriteNodeFn fn);

Nodes shd_rewrite_ops(Rewriter* rewriter, NodeClass class, String op_name, Nodes old_nodes);
Nodes shd_rewrite_ops_with_fn(Rewriter* rewriter, NodeClass class, String op_name, Nodes values, RewriteOpFn fn);

struct Rewriter_ {
    IrArena* src_arena;
    IrArena* dst_arena;
    Module* src_module;
    Module* dst_module;
    RewriteNodeFn rewrite_fn;
    RewriteOpFn rewrite_op_fn;
    struct {
        bool search_map;
        bool write_map;
    } config;

    Rewriter* parent;

    struct Dict* map;
    bool own_decls;
    struct Dict* decls_map;
};

Rewriter shd_create_rewriter_base(Module* src, Module* dst);
Rewriter shd_create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn);
Rewriter shd_create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn);
Rewriter shd_create_importer(Module* src, Module* dst);

Rewriter shd_create_children_rewriter(Rewriter* parent);
Rewriter shd_create_decl_rewriter(Rewriter* parent);
void shd_destroy_rewriter(Rewriter* r);

void shd_rewrite_module(Rewriter* rewriter);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* shd_recreate_node(Rewriter* rewriter, const Node* node);

/// Rewrites a constant / function header
Node* shd_recreate_node_head(Rewriter* rewriter, const Node* old);
void  shd_recreate_node_body(Rewriter* rewriter, const Node* old, Node* new);

/// Rewrites a variable under a new identity
const Node* shd_recreate_param(Rewriter* rewriter, const Node* old);
Nodes shd_recreate_params(Rewriter* rewriter, Nodes oparams);

/// Looks up if the node was already processed
const Node** shd_search_processed(const Rewriter* ctx, const Node* old);
/// Same as shd_search_processed but asserts if it fails to find a mapping
const Node* shd_find_processed(const Rewriter* ctx, const Node* old);
void shd_register_processed(Rewriter* ctx, const Node* old, const Node* new);
void shd_register_processed_list(Rewriter* rewriter, Nodes old, Nodes new);

void shd_dump_rewriter_map(Rewriter* r);

#endif
