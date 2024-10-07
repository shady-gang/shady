#include "shady/pass.h"

#include "../ir_private.h"
#include "../type.h"
#include "../transform/ir_gen_helpers.h"
#include "../analysis/cfg.h"
#include "../analysis/scheduler.h"
#include "../analysis/free_frontier.h"

#include "log.h"
#include "dict.h"
#include "portability.h"

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

typedef struct {
    Rewriter rewriter;
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
            fn_ctx.scheduler = new_scheduler(fn_ctx.cfg);

            Node* new_fn = shd_recreate_node_head(r, node);
            shd_recreate_node_body(&fn_ctx.rewriter, node, new_fn);

            destroy_scheduler(fn_ctx.scheduler);
            destroy_cfg(fn_ctx.cfg);
            return new_fn;
        }
        case BasicBlock_TAG: {
            CFNode* n = cfg_lookup(ctx->cfg, node);
            if (is_cfnode_structural_target(n))
                break;
            struct Dict* frontier = free_frontier(ctx->scheduler, ctx->cfg, node);
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
            Node* new_bb = basic_block(a, new_params, get_abstraction_name_unsafe(node));

            Context* fn_ctx = ctx;
            while (fn_ctx->rewriter.parent) {
                Context* parent_ctx = (Context*) fn_ctx->rewriter.parent;
                if (parent_ctx->cfg)
                    fn_ctx = parent_ctx;
                else
                    break;
            }

            shd_register_processed(&fn_ctx->rewriter, node, new_bb);
            set_abstraction_body(new_bb, shd_rewrite_node(&bb_ctx.rewriter, get_abstraction_body(node)));
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

Module* lift_everything(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    bool todo = true;
    Module* dst;
    while (todo) {
        todo = false;
        dst = new_module(a, get_module_name(src));
        Context ctx = {
            .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process),
            .lift = shd_new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
        };
        shd_rewrite_module(&ctx.rewriter);
        shd_destroy_dict(ctx.lift);
        shd_destroy_rewriter(&ctx.rewriter);
        src = dst;
    }
    return dst;
}
