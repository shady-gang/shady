#ifndef SHADY_REWRITE_H
#define SHADY_REWRITE_H

#include "shady/ir/grammar.h"

typedef struct Rewriter_ Rewriter;

typedef const Node* (*RewriteNodeFn)(Rewriter*, const Node*);
typedef struct { const Node* result; NodeClass mask; } OpRewriteResult;
typedef OpRewriteResult (*RewriteOpFn)(Rewriter*, NodeClass, String, const Node*);

const Node* shd_rewrite_node(Rewriter*, const Node* old);
const Node* shd_rewrite_node_with_fn(Rewriter*, const Node* old, RewriteNodeFn fn);

const Node* shd_rewrite_op(Rewriter*, NodeClass class, String op_name, const Node* old);
const Node* shd_rewrite_op_with_fn(Rewriter*, NodeClass class, String op_name, const Node* old, RewriteOpFn fn);

/// Applies the rewriter to all nodes in the collection
Nodes shd_rewrite_nodes(Rewriter*, Nodes old);
Nodes shd_rewrite_nodes_with_fn(Rewriter*, Nodes old, RewriteNodeFn fn);

Nodes shd_rewrite_ops(Rewriter*, NodeClass class, String op_name, Nodes old);
Nodes shd_rewrite_ops_with_fn(Rewriter*, NodeClass class, String op_name, Nodes old, RewriteOpFn fn);

typedef struct Arena_ Arena;

struct Rewriter_ {
    IrArena* src_arena;
    IrArena* dst_arena;
    Module* src_module;
    Module* dst_module;
    RewriteNodeFn rewrite_fn;
    RewriteOpFn rewrite_op_fn;

    Arena* arena;
    Rewriter* parent;
    struct Dict* map;
};

Rewriter shd_create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn);
Rewriter shd_create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn);
Rewriter shd_create_importer(Module* src, Module* dst);
Rewriter shd_create_children_rewriter(Rewriter* parent);

void shd_destroy_rewriter(Rewriter*);

void shd_rewrite_module(Rewriter*);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* shd_recreate_node(Rewriter*, const Node* node);

/// Rewrites a constant / function header
Node* shd_recreate_node_head(Rewriter*, const Node* old);
void  shd_recreate_node_body(Rewriter*, const Node* old, Node* new);

/// Rewrites a variable under a new identity
const Node* shd_recreate_param(Rewriter*, const Node* oparam);
Nodes shd_recreate_params(Rewriter*, Nodes oparams);

/// Looks up if the node was already processed
const Node** shd_search_processed(const Rewriter*, const Node* old);
const Node** shd_search_processed_mask(const Rewriter*, const Node* old, NodeClass mask);

void shd_register_processed(Rewriter*, const Node* old, const Node* new);
void shd_register_processed_mask(Rewriter*, const Node* old, const Node* new, NodeClass mask);
void shd_register_processed_list(Rewriter*, Nodes old, Nodes new);

Rewriter* shd_get_top_rewriter(Rewriter*);

void shd_dump_rewriter_map(Rewriter*);

#endif
