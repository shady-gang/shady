#include "shady/pass.h"

#include "ir_private.h"
#include "analysis/cfg.h"
#include "analysis/scheduler.h"
#include "analysis/free_frontier.h"

#include "dict.h"
#include "portability.h"

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

typedef struct Context_ {
    Rewriter rewriter;
    struct Context_* fn_ctx;
    struct Dict* lift;
    CFG* cfg;
    Scheduler* scheduler;
} Context;

static const Node* process(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.scheduler = shd_new_scheduler(fn_ctx.cfg);\
            fn_ctx.fn_ctx = &fn_ctx;

            Node* new_fn = shd_recreate_node_head(r, node);
            shd_recreate_node_body(&fn_ctx.rewriter, node, new_fn);

            shd_destroy_scheduler(fn_ctx.scheduler);
            shd_destroy_cfg(fn_ctx.cfg);
            return new_fn;
        }
        case BasicBlock_TAG: {
            CFNode* n = shd_cfg_lookup(ctx->cfg, node);
            if (shd_cfg_is_node_structural_target(n))
                break;
            struct Dict* frontier = shd_free_frontier(ctx->scheduler, ctx->cfg, node);
            // insert_dict(const Node*, Dict*, ctx->lift, node, frontier);

            Nodes additional_args = shd_empty(a);
            Nodes new_params = shd_recreate_params(r, get_abstraction_params(node));
            shd_register_processed_list(r, get_abstraction_params(node), new_params);
            size_t i = 0;
            const Node* value;

            Context bb_ctx = *ctx;
            bb_ctx.rewriter = shd_create_children_rewriter(&ctx->rewriter);

            while (shd_dict_iter(frontier, &i, &value, NULL)) {
                if (is_value(value)) {
                    additional_args = shd_nodes_append(a, additional_args, value);
                    const Type* t = shd_rewrite_node(r, value->type);
                    const Node* p = param(a, t, NULL);
                    new_params = shd_nodes_append(a, new_params, p);
                    shd_register_processed(&bb_ctx.rewriter, value, p);
                }
            }

            shd_destroy_dict(frontier);
            shd_dict_insert(const Node*, Nodes, ctx->lift, node, additional_args);
            Node* new_bb = basic_block(a, new_params, shd_get_abstraction_name_unsafe(node));

            shd_register_processed(&ctx->fn_ctx->rewriter, node, new_bb);
            shd_set_abstraction_body(new_bb, shd_rewrite_node(&bb_ctx.rewriter, get_abstraction_body(node)));
            shd_destroy_rewriter(&bb_ctx.rewriter);
            return new_bb;
        }
        case Jump_TAG: {
            Jump payload = node->payload.jump;
            shd_rewrite_node(r, payload.target);

            Nodes* additional_args = shd_dict_find_value(const Node*, Nodes, ctx->lift, payload.target);
            assert(additional_args);
            return jump(a, (Jump) {
                .mem = shd_rewrite_node(r, payload.mem),
                .target = shd_rewrite_node(r, payload.target),
                .args = shd_concat_nodes(a, shd_rewrite_nodes(r, payload.args), shd_rewrite_nodes(r, *additional_args))
            });
        }
        default: break;
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

Module* shd_pass_lift_everything(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    bool todo = true;
    Module* dst;
    while (todo) {
        todo = false;
        dst = shd_new_module(a, shd_module_get_name(src));
        Context ctx = {
            .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
            .lift = shd_new_dict(const Node*, Nodes, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
        };
        shd_rewrite_module(&ctx.rewriter);
        shd_destroy_dict(ctx.lift);
        shd_destroy_rewriter(&ctx.rewriter);
        src = dst;
    }
    return dst;
}
