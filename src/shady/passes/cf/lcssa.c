#include "ir_private.h"

#include "shady/ir.h"
#include "shady/pass.h"
#include "shady/rewrite.h"

#include "analysis/cfg.h"
#include "analysis/scheduler.h"
#include "analysis/looptree.h"
#include "analysis/free_frontier.h"

#include "portability.h"
#include "log.h"
#include "dict.h"
#include "list.h"

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    const Node* current_fn;
    CFG* cfg;
    Scheduler* scheduler;
    LoopTree* loop_tree;
    struct Dict* lifted_arguments;
} Context;

static bool is_child(const LTNode* maybe_parent, const LTNode* child) {
    const LTNode* n = child;
    while (n) {
        if (n == maybe_parent)
            return true;
        n = n->parent;
    }
    return NULL;
}

static const LTNode* get_loop(const LTNode* n) {
    const LTNode* p = n->parent;
    if (p && p->type == LF_HEAD)
        return p;
    return NULL;
}

static String loop_name(const LTNode* n) {
    if (n && n->type == LF_HEAD && shd_list_count(n->cf_nodes) > 0) {
        return shd_get_node_name_safe(shd_read_list(CFNode*, n->cf_nodes)[0]->node);
    }
    return "fn entry";
}

static void find_liftable_loop_values(Context* ctx, const Node* old, Nodes* nparams, Nodes* lparams) {
    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    assert(old->tag == BasicBlock_TAG);

    const LTNode* bb_loop = get_loop(shd_loop_tree_lookup(ctx->loop_tree, old));

    *nparams = shd_empty(a);
    *lparams = shd_empty(a);

    NodeSet fvs = shd_free_frontier(ctx->scheduler, ctx->cfg, old);
    const Node* fv;
    for (size_t i = 0; shd_node_set_iter(fvs, &i, &fv);) {
        const CFNode* defining_cf_node = shd_schedule_instruction(ctx->scheduler, fv);
        assert(defining_cf_node);
        const LTNode* defining_loop = get_loop(shd_loop_tree_lookup(ctx->loop_tree, defining_cf_node->node));
        if (!is_child(defining_loop, bb_loop)) {
            // that's it, that variable is leaking !
            shd_log_fmt(DEBUGV, "lcssa: ");
            // shd_log_node(DEBUGV, fv);
            shd_log_fmt(DEBUGV, " (%%%d) is used outside of the loop that defines it %s (use in %s)\n", fv->id, loop_name(defining_loop), loop_name(bb_loop));
            const Node* nparam = param_helper(a, shd_rewrite_node(r, fv->type));
            String name = "lcssa_phi";
            String oname = shd_get_node_name_unsafe(fv);
            if (oname)
                name = shd_fmt_string_irarena(a, "lcssa_phi_%s", oname);
            shd_set_debug_name(nparam, name);
            *nparams = shd_nodes_append(a, *nparams, nparam);
            *lparams = shd_nodes_append(a, *lparams, fv);
        }
    }
    shd_destroy_node_set(fvs);
}

static const Node* process_abstraction_body(Context* ctx, const Node* old, const Node* body) {
    IrArena* a = ctx->rewriter.dst_arena;
    Context ctx2 = *ctx;
    ctx = &ctx2;
    Rewriter* r = &ctx->rewriter;

    if (!ctx->cfg) {
        shd_error_print("LCSSA: Trying to process an abstraction that's not part of a function ('%s')!", shd_get_node_name_safe(old));
        shd_log_module(ERROR, ctx->rewriter.src_module);
        shd_error_die();
    }
    const CFNode* n = shd_cfg_lookup(ctx->cfg, old);

    size_t children_count = 0;
    LARRAY(const Node*, old_children, shd_list_count(n->dominates));
    for (size_t i = 0; i < shd_list_count(n->dominates); i++) {
        CFNode* c = shd_read_list(CFNode*, n->dominates)[i];
        if (shd_cfg_is_node_structural_target(c))
            continue;
        old_children[children_count++] = c->node;
    }

    LARRAY(Node*, new_children, children_count);
    LARRAY(Nodes, lifted, children_count);
    LARRAY(Nodes, new_params, children_count);
    for (size_t i = 0; i < children_count; i++) {
        shd_debugv_print("doing %s\n", shd_get_node_name_safe(old_children[i]));
        find_liftable_loop_values(ctx, old_children[i], &new_params[i], &lifted[i]);
        Nodes nparams = shd_recreate_params(&ctx->rewriter, get_abstraction_params(old_children[i]));
        new_children[i] = basic_block_helper(a, shd_concat_nodes(a, nparams, new_params[i]));
        shd_rewrite_annotations(r, old_children[i], new_children[i]);
        shd_register_processed(&ctx->rewriter, old_children[i], new_children[i]);
        shd_register_processed_list(&ctx->rewriter, get_abstraction_params(old_children[i]), nparams);
        assert(!shd_dict_find_key(const Node*, ctx->lifted_arguments, old_children[i]));
        shd_dict_insert(const Node*, Nodes, ctx->lifted_arguments, old_children[i], lifted[i]);
    }

    const Node* new = shd_rewrite_node(&ctx->rewriter, body);

    for (size_t i = 0; i < children_count; i++) {
        Context child_ctx = *ctx;
        child_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);
        shd_register_processed_list(&child_ctx.rewriter, lifted[i], new_params[i]);
        new_children[i]->payload.basic_block.body = process_abstraction_body(&child_ctx, old_children[i], get_abstraction_body(old_children[i]));
        shd_destroy_rewriter(&child_ctx.rewriter);
    }

    return new;
}

static const Node* process_node(Context* ctx, const Node* old) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (old->tag) {
        case NominalType_TAG:
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Context not_a_fn_ctx = *ctx;
            ctx = &not_a_fn_ctx;
            ctx->cfg = NULL;
            return shd_recreate_node(&ctx->rewriter, old);
        }
        case Function_TAG: {
            Context fn_ctx = *ctx;
            ctx = &fn_ctx;

            ctx->current_fn = old;
            ctx->cfg = build_fn_cfg(old);
            ctx->scheduler = shd_new_scheduler(ctx->cfg);
            ctx->loop_tree = shd_new_loop_tree(ctx->cfg);

            Node* new = shd_recreate_node_head(&ctx->rewriter, old);
            new->payload.fun.body = process_abstraction_body(ctx, old, get_abstraction_body(old));

            shd_destroy_loop_tree(ctx->loop_tree);
            shd_destroy_scheduler(ctx->scheduler);
            shd_destroy_cfg(ctx->cfg);
            return new;
        }
        case Jump_TAG: {
            Jump payload = old->payload.jump;
            Nodes nargs = shd_rewrite_nodes(&ctx->rewriter, old->payload.jump.args);
            Nodes* to_lift = shd_dict_find_value(const Node*, Nodes, ctx->lifted_arguments, old->payload.jump.target);
            if (to_lift) {
                Nodes lifted = shd_rewrite_nodes(r, *to_lift);
                nargs = shd_concat_nodes(a, nargs, lifted);
            }
            return jump(a, (Jump) {
                .target = shd_rewrite_node(&ctx->rewriter, old->payload.jump.target),
                .args = nargs,
                .mem = shd_rewrite_node(r, payload.mem),
            });
        }
        case BasicBlock_TAG: {
            assert(false);
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, old);
}

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

Module* shd_pass_lcssa(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .current_fn = NULL,
        .lifted_arguments = shd_new_dict(const Node*, Nodes, (HashFn) shd_hash_node, (CmpFn) shd_compare_node)
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.lifted_arguments);
    return dst;
}
