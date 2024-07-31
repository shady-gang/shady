#include "pass.h"

#include "../type.h"
#include "../ir_private.h"

#include "../analysis/callgraph.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "util.h"
#include "log.h"

typedef struct {
    const Node* host_fn;
    const Node* return_jp;
} InlinedCall;

typedef struct {
    Rewriter rewriter;
    const CompilerConfig* config;
    CallGraph* graph;
    const Node* old_fun;
    Node* fun;
    InlinedCall* inlined_call;
} Context;

static const Node* ignore_immediate_fn_addr(const Node* node) {
    if (node->tag == FnAddr_TAG) {
        return node->payload.fn_addr.fn;
    }
    return node;
}

static bool is_call_potentially_inlineable(const Node* src_fn, const Node* dst_fn) {
    if (lookup_annotation(src_fn, "Internal"))
        return false;
    if (lookup_annotation(dst_fn, "NoInline"))
        return false;
    if (!dst_fn->payload.fun.body)
        return false;
    return true;
}

static bool is_call_safely_removable(const Node* fn) {
    if (lookup_annotation(fn, "Internal"))
        return false;
    if (lookup_annotation(fn, "EntryPoint"))
        return false;
    if (lookup_annotation(fn, "Exported"))
        return false;
    return true;
}

typedef struct {
    size_t num_calls;
    size_t num_inlineable_calls;
    bool can_be_inlined;
    bool can_be_eliminated;
} FnInliningCriteria;

static FnInliningCriteria get_inlining_heuristic(const CompilerConfig* config, CGNode* fn_node) {
    FnInliningCriteria crit = { 0 };

    CGEdge e;
    size_t i = 0;
    while (dict_iter(fn_node->callers, &i, &e, NULL)) {
        crit.num_calls++;
        if (is_call_potentially_inlineable(e.src_fn->fn, e.dst_fn->fn))
            crit.num_inlineable_calls++;
    }

    // a function can be inlined if it has exactly one inlineable call...
    if (crit.num_inlineable_calls <= 1 || config->optimisations.inline_everything)
        crit.can_be_inlined = true;

    // avoid inlining recursive things for now
    if (fn_node->is_address_captured || fn_node->is_recursive)
        crit.can_be_inlined = false;

    // it can be eliminated if it can be inlined, and all the calls are inlineable calls ...
    if (crit.num_calls == crit.num_inlineable_calls && crit.can_be_inlined)
        crit.can_be_eliminated = true;

    // unless the address is captured, in which case it must remain available for the indirect calls.
    if (fn_node->is_address_captured)
        crit.can_be_eliminated = false;

    if (!is_call_safely_removable(fn_node->fn))
        crit.can_be_eliminated = false;

    debugv_print("inlining heuristic for '%s': num_calls=%d num_inlineable_calls=%d safely_removable=%d address_leaks=%d recursive=%d inlineable=%d can_be_eliminated=%d\n",
                 get_abstraction_name(fn_node->fn),
                 crit.num_calls,
                 crit.num_inlineable_calls,
                 is_call_safely_removable(fn_node->fn),
                 fn_node->is_address_captured,
                 fn_node->is_recursive,
                 crit.can_be_inlined,
                 crit.can_be_eliminated);

    return crit;
}

/// inlines the abstraction with supplied arguments
static const Node* inline_call(Context* ctx, const Node* ocallee, Nodes nargs, const Node* return_to) {
    assert(is_abstraction(ocallee));

    log_string(DEBUG, "Inlining '%s' inside '%s'\n", get_abstraction_name(ocallee), get_abstraction_name(ctx->fun));
    Context inline_context = *ctx;
    inline_context.rewriter.map = clone_dict(inline_context.rewriter.map);

    ctx = &inline_context;
    InlinedCall inlined_call = {
        .host_fn = ctx->fun,
        .return_jp = return_to,
    };
    inline_context.inlined_call = &inlined_call;

    Nodes oparams = get_abstraction_params(ocallee);
    register_processed_list(&inline_context.rewriter, oparams, nargs);

    const Node* nbody = rewrite_node(&inline_context.rewriter, get_abstraction_body(ocallee));

    destroy_dict(inline_context.rewriter.map);

    assert(is_terminator(nbody));
    return nbody;
}

static const Node* process(Context* ctx, const Node* node) {
    if (!node)
        return NULL;
    const Node* found = search_processed(&ctx->rewriter, node);
    if (found)
        return found;

    IrArena* a = ctx->rewriter.dst_arena;
    Rewriter* r = &ctx->rewriter;
    assert(a != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    switch (node->tag) {
        case Function_TAG: {
            if (ctx->graph) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
                if (get_inlining_heuristic(ctx->config, fn_node).can_be_eliminated) {
                    debugv_print("Eliminating %s because it has exactly one caller\n", get_abstraction_name(fn_node->fn));
                    return NULL;
                }
            }

            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_params(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            register_processed(r, node, new);

            Context fn_ctx = *ctx;
            fn_ctx.rewriter.map = clone_dict(fn_ctx.rewriter.map);
            fn_ctx.old_fun = node;
            fn_ctx.fun = new;
            fn_ctx.inlined_call = NULL;
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&fn_ctx.rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
            destroy_dict(fn_ctx.rewriter.map);
            return new;
        }
        case Call_TAG: {
            if (!ctx->graph)
                break;
            const Node* ocallee = node->payload.call.callee;
            Nodes oargs = node->payload.call.args;

            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(ctx->config, fn_node).can_be_inlined && is_call_potentially_inlineable(ctx->old_fun, ocallee)) {
                    debugv_print("Inlining call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, oargs);

                    // Prepare a join point to replace the old function return
                    Nodes nyield_types = strip_qualifiers(a, rewrite_nodes(&ctx->rewriter, ocallee->payload.fun.return_types));
                    const Type* jp_type = join_point_type(a, (JoinPointType) { .yield_types = nyield_types });
                    const Node* join_point = param(a, qualified_type_helper(jp_type, true), format_string_arena(a->arena, "inlined_return_%s", get_abstraction_name(ocallee)));

                    const Node* nbody = inline_call(ctx, ocallee, nargs, join_point);

                    BodyBuilder* bb = begin_body(a);
                    return yield_values_and_wrap_in_block(bb, gen_control(bb, nyield_types, case_(a, singleton(join_point), nbody)));
                }
            }
            break;
        }
        case BasicBlock_TAG: {
            Nodes nparams = recreate_params(r, get_abstraction_params(node));
            register_processed_list(r, get_abstraction_params(node), nparams);
            Node* bb = basic_block(a, nparams, get_abstraction_name(node));
            register_processed(r, node, bb);
            bb->payload.basic_block.body = rewrite_node(r, get_abstraction_body(node));
            return bb;
        }
        case Return_TAG: {
            if (ctx->inlined_call)
                return join(a, (Join) { .join_point = ctx->inlined_call->return_jp, .args = rewrite_nodes(r, node->payload.fn_ret.args )});
            break;
        }
        case TailCall_TAG: {
            if (!ctx->graph)
                break;
            const Node* ocallee = node->payload.tail_call.target;
            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(ctx->config, fn_node).can_be_inlined) {
                    debugv_print("Inlining tail call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.tail_call.args);
                    return inline_call(ctx, ocallee, nargs, NULL);
                }
            }
            break;
        }
        default: break;
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

void opt_simplify_cf(const CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process),
        .config = config,
        .graph = NULL,
        .fun = NULL,
        .inlined_call = NULL,
    };
    ctx.graph = new_callgraph(src);

    rewrite_module(&ctx.rewriter);
    if (ctx.graph)
        destroy_callgraph(ctx.graph);

    destroy_rewriter(&ctx.rewriter);
}

Module* opt_inline(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    opt_simplify_cf(config, src, dst);
    return dst;
}
