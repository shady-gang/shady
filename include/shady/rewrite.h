#ifndef SHADY_REWRITE_H
#define SHADY_REWRITE_H

#include "shady/ir/grammar.h"

typedef struct Rewriter_ Rewriter;

typedef const Node* (*RewriteNodeFn)(Rewriter*, const Node*);

typedef struct OpRewriteResult_ OpRewriteResult;
typedef OpRewriteResult* (*RewriteOpFn)(Rewriter*, NodeClass, String, const Node*);

OpRewriteResult* shd_new_rewrite_result(Rewriter*, const Node* defaultResult);
OpRewriteResult* shd_new_rewrite_result_none(Rewriter*);
void shd_rewrite_result_add_mask_rule(OpRewriteResult*, NodeClass mask, const Node* defaultResult);

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

typedef Rewriter* (SelectRewriterFn)(Rewriter*, const Node*);

struct Rewriter_ {
    IrArena* src_arena;
    IrArena* dst_arena;
    Module* src_module;
    Module* dst_module;
    RewriteNodeFn rewrite_fn;
    RewriteOpFn rewrite_op_fn;
    SelectRewriterFn* select_rewriter_fn;

    Arena* arena;
    Rewriter* parent;
    struct Dict* map;
};

SelectRewriterFn shd_default_rewriter_selector;

Rewriter shd_create_node_rewriter(Module* src, Module* dst, RewriteNodeFn fn);
Rewriter shd_create_op_rewriter(Module* src, Module* dst, RewriteOpFn fn);
Rewriter shd_create_importer(Module* src, Module* dst);
Rewriter shd_create_children_rewriter(Rewriter* parent);

void shd_destroy_rewriter(Rewriter*);

void shd_rewrite_module(Rewriter*);

/// Rewrites a node using the rewriter to provide the node and type operands
const Node* shd_recreate_node(Rewriter*, const Node* old);

/// Rewrites a constant / function header
Node* shd_recreate_node_head_(Rewriter*, const Node* old);
Node* shd_recreate_node_head(Rewriter*, const Node* old);
void  shd_recreate_node_body(Rewriter*, const Node* old, Node* new);

Function shd_rewrite_function_head_payload(Rewriter* r, Function old);
GlobalVariable shd_rewrite_global_head_payload(Rewriter* r, GlobalVariable old);
Constant shd_rewrite_constant_head_payload(Rewriter* r, Constant old);
StructType shd_rewrite_struct_type_head_payload(Rewriter* r, StructType old);
BasicBlock shd_rewrite_basic_block_head_payload(Rewriter* r, BasicBlock old);

/// Rewrites a variable under a new identity
const Node* shd_recreate_param(Rewriter*, const Node* oparam);
Nodes shd_recreate_params(Rewriter*, Nodes oparams);

void shd_rewrite_annotations(Rewriter* r, const Node* old, Node* new_);

/// Looks up if the node was already processed
const Node* shd_search_processed(const Rewriter*, const Node* old);
const Node* shd_search_processed_by_use_class(const Rewriter*, const Node* old, NodeClass use);

void shd_register_processed(Rewriter*, const Node* old, const Node* new);
void shd_register_processed_result(Rewriter*, const Node* old, const OpRewriteResult*);
void shd_register_processed_list(Rewriter*, Nodes old, Nodes new);

Rewriter* shd_get_top_rewriter(Rewriter*);

void shd_dump_rewriter_map(Rewriter*);

#endif
