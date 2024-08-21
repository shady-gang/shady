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

            Node* new_fn = recreate_decl_header_identity(r, node);
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new_fn);

            destroy_scheduler(fn_ctx.scheduler);
            destroy_cfg(fn_ctx.cfg);
            return new_fn;
        }
        case BasicBlock_TAG: {
            struct Dict* frontier = free_frontier(ctx->scheduler, ctx->cfg, node);
            // insert_dict(const Node*, Dict*, ctx->lift, node, frontier);

            Nodes additional_args = empty(a);
            Nodes new_params = recreate_params(r, get_abstraction_params(node));
            register_processed_list(r, get_abstraction_params(node), new_params);
            size_t i = 0;
            const Node* value;

            Context bb_ctx = *ctx;
            bb_ctx.rewriter = create_children_rewriter(&ctx->rewriter);

            while (dict_iter(frontier, &i, &value, NULL)) {
                if (is_value(value)) {
                    additional_args = append_nodes(a, additional_args, value);
                    const Type* t = rewrite_node(r, value->type);
                    const Node* p = param(a, t, NULL);
                    new_params = append_nodes(a, new_params, p);
                    register_processed(&bb_ctx.rewriter, value, p);
                }
            }

            destroy_dict(frontier);
            insert_dict(const Node*, Nodes, ctx->lift, node, additional_args);
            Node* new_bb = basic_block(a, new_params, get_abstraction_name_unsafe(node));
            register_processed(r, node, new_bb);
            set_abstraction_body(new_bb, rewrite_node(&bb_ctx.rewriter, get_abstraction_body(node)));
            return new_bb;
        }
        case Jump_TAG: {
            Jump payload = node->payload.jump;
            rewrite_node(r, payload.target);

            Nodes* additional_args = find_value_dict(const Node*, Nodes, ctx->lift, payload.target);
            assert(additional_args);
            return jump(a, (Jump) {
                .mem = rewrite_node(r, payload.mem),
                .target = rewrite_node(r, payload.target),
                .args = concat_nodes(a, rewrite_nodes(r, payload.args), rewrite_nodes(r, *additional_args))
            });
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

Module* lift_everything(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .lift = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_dict(ctx.lift);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
