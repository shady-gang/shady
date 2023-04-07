#include "passes.h"

#include "dict.h"
#include "list.h"
#include "portability.h"
#include "log.h"

#include "../rewrite.h"
#include "../type.h"

#include "../analysis/scope.h"
#include "../analysis/callgraph.h"

typedef struct {
    Rewriter rewriter;
    Scope* scope;
    CallGraph* graph;
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

static bool is_call_potentially_inlineable(CGEdge edge) {
    if (lookup_annotation(edge.src_fn->fn, "Leaf"))
        return false;
    if (lookup_annotation(edge.dst_fn->fn, "NoInline"))
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
        if (is_call_potentially_inlineable(e))
            crit.num_inlineable_calls++;
    }

    // a function can be inlined if it has exactly one inlineable call...
    if (crit.num_inlineable_calls == 1)
        crit.can_be_inlined = true;

    // avoid inlining recursive things
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
static const Node* inline_call(Context* ctx, const Node* oabs, Nodes nargs) {
    assert(is_abstraction(oabs));

    Context inline_context = *ctx;
    inline_context.rewriter.map = clone_dict(inline_context.rewriter.map);
    Nodes oparams = get_abstraction_params(oabs);
    register_processed_list(&inline_context.rewriter, oparams, nargs);

    if (oabs->tag == Function_TAG)
        inline_context.scope = new_scope(oabs);

    const Node* nbody = rewrite_node(&inline_context.rewriter, get_abstraction_body(oabs));

    if (oabs->tag == Function_TAG)
        destroy_scope(inline_context.scope);

    destroy_dict(inline_context.rewriter.map);

    assert(is_terminator(nbody));
    return nbody;
}

static const Node* process(Context* ctx, const Node* node) {
    IrArena* arena = ctx->rewriter.dst_arena;
    if (!node) return NULL;
    assert(arena != node->arena);
    assert(node->arena == ctx->rewriter.src_arena);

    const Node* found = search_processed(&ctx->rewriter, node);
    if (found) return found;

    switch (node->tag) {
        case Function_TAG: {
            CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, node);
            if (get_inlining_heuristic(fn_node).can_be_eliminated) {
                debugv_print("Eliminating %s because it has exactly one caller\n", get_abstraction_name(fn_node->fn));
                return NULL;
            }

            Nodes annotations = rewrite_nodes(&ctx->rewriter, node->payload.fun.annotations);
            Node* new = function(ctx->rewriter.dst_module, recreate_variables(&ctx->rewriter, node->payload.fun.params), node->payload.fun.name, annotations, rewrite_nodes(&ctx->rewriter, node->payload.fun.return_types));
            for (size_t i = 0; i < new->payload.fun.params.count; i++)
                register_processed(&ctx->rewriter, node->payload.fun.params.nodes[i], new->payload.fun.params.nodes[i]);
            register_processed(&ctx->rewriter, node, new);

            Context fn_ctx = *ctx;
            Scope* scope = new_scope(node);
            fn_ctx.scope = scope;
            fn_ctx.fun = new;
            recreate_decl_body_identity(&fn_ctx.rewriter, node, new);
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
                debugv_print("Inlining jump to %s\n", get_abstraction_name(otarget));
                Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.jump.args);
                return inline_call(ctx, otarget, nargs);
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case IndirectCall_TAG:
        case LeafCall_TAG: {
            const Node* ocallee = node->tag == LeafCall_TAG ? node->payload.leaf_call.callee : node->payload.indirect_call.callee;
            Nodes oargs = node->tag == LeafCall_TAG ? node->payload.leaf_call.args : node->payload.indirect_call.args;

            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(fn_node).can_be_inlined) {
                    debugv_print("Inlining call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, oargs);

                    // Prepare a join point to replace the old function return
                    Nodes nyield_types = strip_qualifiers(arena, rewrite_nodes(&ctx->rewriter, ocallee->payload.fun.return_types));
                    const Type* jp_type = join_point_type(arena, (JoinPointType) { .yield_types = nyield_types });
                    const Node* join_point = var(arena, qualified_type_helper(jp_type, true), format_string(arena, "inlined_return_%s", get_abstraction_name(ocallee)));
                    insert_dict_and_get_result(const Node*, const Node*, ctx->inlined_return_sites, ocallee, join_point);

                    const Node* nbody = inline_call(ctx, ocallee, nargs);

                    remove_dict(const Node*, ctx->inlined_return_sites, ocallee);

                    return control(arena, (Control) {
                        .yield_types = nyield_types,
                        .inside = lambda(arena, singleton(join_point), nbody)
                    });
                }
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case Return_TAG: {
            const Node** p_ret_jp = find_value_dict(const Node*, const Node*, ctx->inlined_return_sites, node);
            if (p_ret_jp)
                return join(arena, (Join) { .join_point = *p_ret_jp, .args = rewrite_nodes(&ctx->rewriter, node->payload.fn_ret.args )});
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case TailCall_TAG: {
            const Node* ocallee = node->payload.tail_call.target;
            ocallee = ignore_immediate_fn_addr(ocallee);
            if (ocallee->tag == Function_TAG) {
                CGNode* fn_node = *find_value_dict(const Node*, CGNode*, ctx->graph->fn2cgn, ocallee);
                if (get_inlining_heuristic(fn_node).can_be_inlined) {
                    debugv_print("Inlining tail call to %s\n", get_abstraction_name(ocallee));
                    Nodes nargs = rewrite_nodes(&ctx->rewriter, node->payload.tail_call.args);

                    return inline_call(ctx, ocallee, nargs);
                }
            }
            return recreate_node_identity(&ctx->rewriter, node);
        }
        case BasicBlock_TAG: {
            Nodes params = recreate_variables(&ctx->rewriter, node->payload.basic_block.params);
            register_processed_list(&ctx->rewriter, node->payload.basic_block.params, params);
            Node* bb = basic_block(arena, (Node*) ctx->fun, params, node->payload.basic_block.name);
            register_processed(&ctx->rewriter, node, bb);
            bb->payload.basic_block.body = process(ctx, node->payload.basic_block.body);
            return bb;
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

void opt_simplify_cf(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .graph = new_callgraph(src),
        .scope = NULL,
        .fun = NULL,
        .inlined_return_sites = new_dict(const Node*, CGNode*, (HashFn) hash_node, (CmpFn) compare_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_callgraph(ctx.graph);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.inlined_return_sites);
}
