#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "util.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"
#include "../ir_private.h"

#include "../analysis/scope.h"
#include "../analysis/callgraph.h"

typedef struct {
    Rewriter rewriter;
    Scope* scope;
    CallGraph* graph;
    const Node* old_fun;
    Node* fun;
    bool allow_fn_inlining;
    struct Dict* inlined_return_sites;
} Context;

static const Node* ignore_immediate_fn_addr(const Node* node) {
    if (node->tag == FnAddr_TAG) {
        return node->payload.fn_addr.fn;
    }
    return node;
}

static bool is_call_potentially_inlineable(const Node* src_fn, const Node* dst_fn) {
    if (lookup_annotation(src_fn, "Leaf"))
        return false;
    if (lookup_annotation(dst_fn, "NoInline"))
        return false;
    return true;
}

typedef struct {
    size_t num_calls;
    size_t num_inlineable_calls;
    bool can_be_inlined;
    bool can_be_eliminated;
} FnInliningCriteria;

static FnInliningCriteria get_inlining_heuristic(CGNode* fn_node) {
    FnInliningCriteria crit = { 0 };

    CGEdge e;
    size_t i = 0;
    while (dict_iter(fn_node->callers, &i, &e, NULL)) {
        crit.num_calls++;
        if (is_call_potentially_inlineable(e.src_fn->fn, e.dst_fn->fn))
            crit.num_inlineable_calls++;
    }

    debugv_print("%s has %d callers\n", get_abstraction_name(fn_node->fn), crit.num_calls);

    // a function can be inlined if it has exactly one inlineable call...
    if (crit.num_inlineable_calls == 1)
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

    return crit;
}

/// inlines the abstraction with supplied arguments
static const Node* inline_call(Context* ctx, const Node* oabs, Nodes nargs, bool separate_scope) {
    assert(is_abstraction(oabs));

    Context inline_context = *ctx;
    if (separate_scope)
        inline_context.rewriter.map = clone_dict(inline_context.rewriter.map);
    Nodes oparams = get_abstraction_params(oabs);
    register_processed_list(&inline_context.rewriter, oparams, nargs);

    if (oabs->tag == Function_TAG)
        inline_context.scope = new_scope(oabs);

    const Node* nbody = rewrite_node(&inline_context.rewriter, get_abstraction_body(oabs));

    if (oabs->tag == Function_TAG)
        destroy_scope(inline_context.scope);

    if (separate_scope)
        destroy_dict(inline_context.rewriter.map);

    assert(is_terminator(nbody));
    return nbody;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* a = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    assert(a != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            if (ctx->graph) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
                if (get_inlining_heuristic(fn_node).can_be_eliminated) {
                    debugv_print("Eliminating %s because it has exactly one caller\n", get_abstraction_name(fn_node->fn));
                    return NULL;
                }
            }

            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            Context fn_ctx = *ctx;
            Scope* scope = new_scope(node);
            fn_ctx.rewriter.map = clone_dict(fn_ctx.rewriter.map);
            fn_ctx.scope = scope;
            fn_ctx.old_fun = node;
            fn_ctx.fun = new;
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
            destroy_dict(fn_ctx.rewriter.map);
            destroy_scope(scope);
            return new;
        }
        case Jump_TAG: {
            const Node* otarget = node->payload.jump.target;
            assert(otarget && otarget->tag == BasicBlock_TAG);
            assert(otarget->payload.basic_block.fn == ctx->scope->entry->node);
            CFNode* cfnode = scope_lookup(ctx->scope, otarget);
            assert(cfnode);
            size_t preds_count = entries_count_list(cfnode->pred_edges);
            assert(preds_count > 0 && "this CFG looks broken");
            if (preds_count == 1) {
                debugv_print("Inlining jump to %s inside function %s\n", get_abstraction_name(otarget), get_abstraction_name(ctx->old_fun));
                Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.jump.args);
                return inline_call(ctx, otarget, nargs, false);
            }
            break;
        }
        // do not inline jumps in branches
        case Branch_TAG: {
            return branch(a, (Branch) {
                .branch_condition = rewrite_node(&ctx->rewriter, node->payload.branch.branch_condition),
                .true_jump = recreate_node_identity(&ctx->rewriter, node->payload.branch.true_jump),
                .false_jump = recreate_node_identity(&ctx->rewriter, node->payload.branch.false_jump),
            });
        }
        case Switch_TAG: {
            return br_switch(a, (Switch) {
                .switch_value = rewrite_node(&ctx->rewriter, node->payload.br_switch.switch_value),
                .case_values = rewrite_nodes(&ctx->rewriter, node->payload.br_switch.case_values),
                .case_jumps = rewrite_nodes_with_fn(&ctx->rewriter, node->payload.br_switch.case_jumps, recreate_node_identity),
                .default_jump = recreate_node_identity(&ctx->rewriter, node->payload.br_switch.default_jump),
            });
        }
        case Call_TAG: {
            if (!ctx->graph)
                break;
            const Node* ocallee = node->payload.call.callee;
            Nodes oargs = node->payload.call.args;

            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(fn_node).can_be_inlined && is_call_potentially_inlineable(ctx->old_fun, ocallee)) {
                    debugv_print("Inlining call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, oargs);

                    // Prepare a join point to replace the old function return
                    Nodes nyield_types = strip_qualifiers(a, rewrite_nodes(&ctx->rewriter, ocallee->payload.fun.return_types));
                    const Type* jp_type = join_point_type(a, (JoinPointType) { .yield_types = nyield_types });
                    const Node* join_point = var(a, qualified_type_helper(jp_type, true), format_string(a->arena, "inlined_return_%s", get_abstraction_name(ocallee)));
                    insert_dict_and_get_result(const Node*, const Node*, ctx->inlined_return_sites, ocallee, join_point);

                    const Node* nbody = inline_call(ctx, ocallee, nargs, true);

                    remove_dict(const Node*, ctx->inlined_return_sites, ocallee);

                    return control(a, (Control) {
                        .yield_types = nyield_types,
                        .inside = lambda(a, singleton(join_point), nbody)
                    });
                }
            }
            break;
        }
        case Return_TAG: {
            const Node** p_ret_jp = find_value_dict(const Node*, const Node*, ctx->inlined_return_sites, node);
            if (p_ret_jp)
                return join(a, (Join) { .join_point = *p_ret_jp, .args = rewrite_nodes(&ctx->rewriter, node->payload.fn_ret.args )});
            break;
        }
        case TailCall_TAG: {
            if (!ctx->graph)
                break;
            const Node* ocallee = node->payload.tail_call.target;
            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(fn_node).can_be_inlined) {
                    debugv_print("Inlining tail call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.tail_call.args);

                    return inline_call(ctx, ocallee, nargs, true);
                }
            }
            break;
        }
        case BasicBlock_TAG: {
            Nodes params = recreate_variables(&ctx->rewriter, node->payload.basic_block.params);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, params);
            Node* bb = basic_block(a, (Node*) ctx->fun, params, node->payload.basic_block.name);
            register_processed(&ctx->rewriter, node, bb);
            bb->payload.basic_block.body = process(ctx, node->payload.basic_block.body);
            return bb;
        }
        default: break;
    }

    const Node* new = recreate_node_identity(&ctx->rewriter, node);
    if (node->tag == AnonLambda_TAG)
        register_processed(&ctx->rewriter, node, new);
    return new;
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

void opt_simplify_cf(SHADY_UNUSED const CompilerConfig* config, Module* src, Module* dst, bool allow_fn_inlining) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .graph = NULL,
        .scope = NULL,
        .fun = NULL,
        .inlined_return_sites = new_dict(const Node*, CGNode*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    if (allow_fn_inlining)
        ctx.graph = new_callgraph(src);

    rewrite_module(&ctx.rewriter);
    if (ctx.graph)
        destroy_callgraph(ctx.graph);

    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.inlined_return_sites);
}

Module* opt_inline_jumps(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    opt_simplify_cf(config, src, dst, false);
    return dst;
}

Module* opt_inline(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));
    opt_simplify_cf(config, src, dst, true);
    return dst;
}
