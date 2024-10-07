#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../analysis/cfg.h"
#include "../analysis/looptree.h"
#include "../transform/ir_gen_helpers.h"

#include "log.h"
#include "list.h"
#include "portability.h"

typedef struct {
    Rewriter rewriter;
    CFG* cfg;
    Nodes* depth_per_rpo;
} Context;

static Nodes to_ids(IrArena* a, Nodes in) {
    LARRAY(const Node*, arr, in.count);
    for (size_t i = 0; i < in.count; i++)
        arr[i] = shd_uint32_literal(a, in.nodes[i]->id);
    return shd_nodes(a, in.count, arr);
}

static void visit_looptree_prepend(IrArena* a, Nodes* arr, LTNode* node, Nodes prefix) {
    if (node->type == LF_HEAD) {
        for (size_t i = 0; i < shd_list_count(node->lf_children); i++) {
            LTNode* n = shd_read_list(LTNode*, node->lf_children)[i];
            visit_looptree_prepend(a, arr, n, prefix);
        }
    } else {
        for (size_t i = 0; i < shd_list_count(node->cf_nodes); i++) {
            CFNode* n = shd_read_list(CFNode*, node->cf_nodes)[i];
            arr[n->rpo_index] = shd_concat_nodes(a, prefix, arr[n->rpo_index]);
        }
        assert(node->lf_children);
    }
}

static bool is_nested(LTNode* a, LTNode* in) {
    assert(a->type == LF_HEAD && in->type == LF_HEAD);
    while (a) {
        if (a == in)
            return true;
        a = a->parent;
    }
    return false;
}

static void paint_dominated_up_to_postdom(CFNode* n, IrArena* a, Nodes* arr, const Node* postdom, const Node* prefix) {
    if (n->node == postdom)
        return;

    for (size_t i = 0; i < shd_list_count(n->dominates); i++) {
        CFNode* dominated = shd_read_list(CFNode*, n->dominates)[i];
        paint_dominated_up_to_postdom(dominated, a, arr, postdom, prefix);
    }

    arr[n->rpo_index] = shd_nodes_prepend(a, arr[n->rpo_index], prefix);
}

static void visit_acyclic_cfg_domtree(CFNode* n, IrArena* a, Nodes* arr, CFG* flipped, LTNode* loop, LoopTree* lt) {
    LTNode* ltn = looptree_lookup(lt, n->node);
    if (ltn->parent != loop)
        return;

    for (size_t i = 0; i < shd_list_count(n->dominates); i++) {
        CFNode* dominated = shd_read_list(CFNode*, n->dominates)[i];
        visit_acyclic_cfg_domtree(dominated, a, arr, flipped, loop, lt);
    }

    CFNode* src = n;

    if (shd_list_count(src->succ_edges) < 2)
        return; // no divergence, no bother

    CFNode* f_src = cfg_lookup(flipped, src->node);
    CFNode* f_src_ipostdom = f_src->idom;
    if (!f_src_ipostdom)
        return;

    // your post-dominator can't be yourself... can it ?
    assert(f_src_ipostdom->node != src->node);

    LTNode* src_lt = looptree_lookup(lt, src->node);
    LTNode* pst_lt = looptree_lookup(lt, f_src_ipostdom->node);
    assert(src_lt->type == LF_LEAF && pst_lt->type == LF_LEAF);
    if (src_lt->parent == pst_lt->parent) {
        shd_log_fmt(DEBUGVV, "We have a candidate for reconvergence: a branch starts at %d and ends at %d\n", src->node->id, f_src_ipostdom->node->id);
        paint_dominated_up_to_postdom(n, a, arr, f_src_ipostdom->node, n->node);
    }
}

static void visit_looptree(IrArena* a, Nodes* arr, const Node* fn, CFG* flipped, LoopTree* lt, LTNode* node) {
    if (node->type == LF_HEAD) {
        Nodes surrounding = shd_empty(a);
        bool is_loop = false;
        for (size_t i = 0; i < shd_list_count(node->cf_nodes); i++) {
            CFNode* n = shd_read_list(CFNode*, node->cf_nodes)[i];
            surrounding = shd_nodes_append(a, surrounding, n->node);
            is_loop = true;
        }

        for (size_t i = 0; i < shd_list_count(node->lf_children); i++) {
            LTNode* n = shd_read_list(LTNode*, node->lf_children)[i];
            visit_looptree(a, arr, fn, flipped, lt, n);
        }

        assert(shd_list_count(node->cf_nodes) < 2);
        CFG* sub_cfg = build_cfg(fn, is_loop ? shd_read_list(CFNode*, node->cf_nodes)[0]->node : fn, (CFGBuildConfig) {
            .include_structured_tails = true,
            .lt = lt
        });

        visit_acyclic_cfg_domtree(sub_cfg->entry, a, arr, flipped, node, lt);

        if (is_loop > 0)
            surrounding = shd_nodes_prepend(a, surrounding, string_lit_helper(a, unique_name(a, "loop_body")));

        visit_looptree_prepend(a, arr, node, surrounding);
        // Remove one level of scoping for the loop headers (forcing reconvergence)
        for (size_t i = 0; i < shd_list_count(node->cf_nodes); i++) {
            CFNode* n = shd_read_list(CFNode*, node->cf_nodes)[i];
            Nodes old = arr[n->rpo_index];
            assert(old.count > 1);
            arr[n->rpo_index] = shd_nodes(a, old.count - 1, &old.nodes[0]);
        }

        destroy_cfg(sub_cfg);
    }
}

static bool loop_depth(LTNode* a) {
    int i = 0;
    while (a) {
        if (shd_list_count(a->cf_nodes) > 0)
           i++;
        else {
            assert(!a->parent);
        }
        a = a->parent;
    }
    return i;
}

static Nodes* compute_scope_depth(IrArena* a, CFG* cfg) {
    CFG* flipped = build_fn_cfg_flipped(cfg->entry->node);
    LoopTree* lt = build_loop_tree(cfg);

    Nodes* arr = calloc(sizeof(Nodes), cfg->size);
    for (size_t i = 0; i < cfg->size; i++)
        arr[i] = shd_empty(a);

    visit_looptree(a, arr, cfg->entry->node, flipped, lt, lt->root);

    // we don't want to cause problems by holding onto pointless references...
    for (size_t i = 0; i < cfg->size; i++)
        arr[i] = to_ids(a, arr[i]);

    destroy_loop_tree(lt);
    destroy_cfg(flipped);

    return arr;
}

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.depth_per_rpo = compute_scope_depth(a, fn_ctx.cfg);
            Node* new_fn = shd_recreate_node_head(r, node);
            BodyBuilder* bb = begin_body_with_mem(a, shd_get_abstraction_mem(new_fn));
            gen_ext_instruction(bb, "shady.scope", 0, unit_type(a), shd_empty(a));
            shd_register_processed(r, shd_get_abstraction_mem(node), bb_mem(bb));
            shd_set_abstraction_body(new_fn, finish_body(bb, shd_rewrite_node(&fn_ctx.rewriter, get_abstraction_body(node))));
            destroy_cfg(fn_ctx.cfg);
            free(fn_ctx.depth_per_rpo);
            return new_fn;
        }
        case BasicBlock_TAG: {
            Nodes nparams = shd_recreate_params(r, get_abstraction_params(node));
            shd_register_processed_list(r, get_abstraction_params(node), nparams);
            Node* new_bb = basic_block(a, nparams, shd_get_abstraction_name_unsafe(node));
            shd_register_processed(r, node, new_bb);
            BodyBuilder* bb = begin_body_with_mem(a, shd_get_abstraction_mem(new_bb));
            CFNode* n = cfg_lookup(ctx->cfg, node);
            gen_ext_instruction(bb, "shady.scope", 0, unit_type(a), ctx->depth_per_rpo[n->rpo_index]);
            shd_register_processed(r, shd_get_abstraction_mem(node), bb_mem(bb));
            shd_set_abstraction_body(new_bb, finish_body(bb, shd_rewrite_node(r, get_abstraction_body(node))));
            return new_bb;
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* scope_heuristic(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    return dst;
}
