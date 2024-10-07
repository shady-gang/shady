#include "shady/pass.h"

#include "../analysis/callgraph.h"
#include "../analysis/cfg.h"
#include "../analysis/uses.h"
#include "../analysis/leak.h"

#include "dict.h"
#include "portability.h"
#include "log.h"

typedef struct {
    Rewriter rewriter;
    CallGraph* graph;
    struct Dict* fns;

    bool is_leaf;
    CFG* cfg;
    const UsesMap* uses;
} Context;

typedef struct {
    CGNode* node;
    // Whether this is a leaf node according to the callgraph
    bool is_leaf;
    bool done;
} FnInfo;

static bool is_leaf_fn(Context* ctx, CGNode* fn_node) {
    FnInfo* info = shd_dict_find_value(const Node*, FnInfo, ctx->fns, fn_node->fn);
    if (info) {
        // if we encounter a function before 'done' is set, it must be part of a recursive chain
        if (!info->done) {
            info->is_leaf = false;
            info->done = true;
        }
        return info->is_leaf;
    }

    FnInfo initial_info = {
        .node = fn_node,
        .done = false,
    };
    shd_dict_insert(const Node*, FnInfo, ctx->fns, fn_node->fn, initial_info);
    info = shd_dict_find_value(const Node*, FnInfo, ctx->fns, fn_node->fn);
    assert(info);

    if (fn_node->is_address_captured || fn_node->is_recursive || fn_node->calls_indirect) {
        info->is_leaf = false;
        info->done = true;
        shd_debugv_print("Function %s can't be a leaf function because", get_abstraction_name(fn_node->fn));
        bool and = false;
        if (fn_node->is_address_captured) {
            shd_debugv_print("its address is captured");
            and = true;
        }
        if (fn_node->is_recursive) {
            if (and)
                shd_debugv_print(" and ");
            shd_debugv_print("it is recursive");
            and = true;
        }
        if (fn_node->calls_indirect) {
            if (and)
                shd_debugv_print(" and ");
            shd_debugv_print("it makes indirect calls");
            and = true;
        }
        shd_debugv_print(".\n");
        return false;
    }

    size_t iter = 0;
    CGEdge e;
    while (shd_dict_iter(fn_node->callees, &iter, &e, NULL)) {
        if (!is_leaf_fn(ctx, e.dst_fn)) {
            shd_debugv_print("Function %s can't be a leaf function because its callee %s is not a leaf function.\n", get_abstraction_name(fn_node->fn), get_abstraction_name(e.dst_fn->fn));
            info->is_leaf = false;
            info->done = true;
        }
    }

    // by analysing the callees, the dict might have been regrown so we must refetch this to update the ptr if needed
    info = shd_dict_find_value(const Node*, FnInfo, ctx->fns, fn_node->fn);

    if (!info->done) {
        info->is_leaf = true;
        info->done = true;
    }

    return info->is_leaf;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            CGNode* fn_node = *shd_dict_find_value(const Node*, CGNode*, ctx->graph->fn2cgn, node);
            fn_ctx.is_leaf = is_leaf_fn(ctx, fn_node);
            fn_ctx.cfg = build_fn_cfg(node);
            fn_ctx.uses = create_fn_uses_map(node, (NcDeclaration | NcType));
            ctx = &fn_ctx;

            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_params(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);
            recreate_decl_body_identity(&ctx->rewriter, node, new);

            if (fn_ctx.is_leaf) {
                shd_debugv_print("Function %s is a leaf function!\n", get_abstraction_name(node));
                new->payload.fun.annotations = shd_nodes_append(a, annotations, annotation(a, (Annotation) {
                    .name = "Leaf",
                }));
            }

            destroy_uses_map(fn_ctx.uses);
            destroy_cfg(fn_ctx.cfg);
            return new;
        }
        case Control_TAG: {
            if (!is_control_static(ctx->uses, node)) {
                shd_debugv_print("Function %s can't be a leaf function because the join point ", get_abstraction_name(ctx->cfg->entry->node));
                shd_log_node(DEBUGV, shd_first(get_abstraction_params(node->payload.control.inside)));
                shd_debugv_print("escapes its control block, preventing restructuring.\n");
                ctx->is_leaf = false;
            }
            break;
        }
        case Join_TAG: {
            const Node* old_jp = node->payload.join.join_point;
            if (old_jp->tag == Param_TAG) {
                const Node* control = get_control_for_jp(ctx->uses, old_jp);
                if (control && is_control_static(ctx->uses, control))
                    break;
            }
            shd_debugv_print("Function %s can't be a leaf function because it joins with ", get_abstraction_name(ctx->cfg->entry->node));
            shd_log_node(DEBUGV, old_jp);
            shd_debugv_print("which is not bound by a control node within that function.\n");
            ctx->is_leaf = false;
            break;
        }
        default:
            break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

Module* mark_leaf_functions(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(get_module_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .fns = shd_new_dict(const Node*, FnInfo, (HashFn) hash_node, (CmpFn) compare_node),
        .graph = new_callgraph(src)
    };
    rewrite_module(&ctx.rewriter);
    shd_destroy_dict(ctx.fns);
    destroy_callgraph(ctx.graph);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
