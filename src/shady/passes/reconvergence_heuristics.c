#include "shady/ir.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"

#include "../type.h"
#include "../rewrite.h"

#include "../analysis/scope.h"
#include "../analysis/looptree.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const Node* current_fn;
    const Node* current_abstraction;
    Scope* fwd_scope;
    Scope* back_scope;
    LoopTree* current_looptree;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    if (node == NULL) return NULL;

    Rewriter * rewriter = &ctx->rewriter;
    IrArena * arena = rewriter->dst_arena;

    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    if (is_function(node)) {
        ctx->current_fn = node;
    }

    if (!ctx->current_fn || !lookup_annotation(ctx->current_fn, "Restructure")) {
        debugv_print("No restructuring here\n");
        return recreate_node_identity(&ctx->rewriter, node);
    }

    if (is_function(node)) {
        if (ctx->fwd_scope) {
            destroy_scope(ctx->fwd_scope);
            ctx->fwd_scope = NULL;
        }
        if (ctx->back_scope) {
            destroy_scope(ctx->back_scope);
            ctx->back_scope = NULL;
        }
    }

    const Node* old_abstraction = ctx->current_abstraction;
    if (is_abstraction(node)) {
        ctx->current_abstraction = node;
    }
    const Node* result;

    if (node->tag == Branch_TAG) {
        if (!ctx->fwd_scope) {
            ctx->fwd_scope = new_scope(ctx->current_fn);
            ctx->back_scope = new_scope_flipped(ctx->current_fn);
            ctx->current_looptree = build_loop_tree(ctx->fwd_scope);
        }

        CFNode* cfnode = scope_lookup(ctx->back_scope, ctx->current_abstraction);
        CFNode* idom = cfnode->idom;

        if(!idom->node) {
            error("Degenerate case: there is no immediate post dominator for this branch.");
        }

        LTNode* lt_node = looptree_lookup(ctx->current_looptree, ctx->current_abstraction);
        CFNode* current_node = scope_lookup(ctx->fwd_scope, ctx->current_abstraction);

        assert(lt_node);
        assert(current_node);

        bool leaves_loop = false;

        for (size_t i = 0; i < entries_count_list(current_node->succ_edges); i++) {
            CFEdge edge = read_list(CFEdge, current_node->succ_edges)[i];
            LTNode* lt_target = looptree_lookup(ctx->current_looptree, edge.dst->node);

            if (lt_target->parent != lt_node->parent) {
                //TODO: I do not trust ipostdom to be correct. Seems to work for simple examples though.
                assert(edge.dst->node == idom->node);
                assert(lt_target->parent->depth == lt_node->parent->depth - 1 && "only one break at a time RN");

                leaves_loop = true;
                break;
            }
        }

        Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

        if (leaves_loop) {
            //Branches leaving loops are not handled here. Recreate a direct copy.
            result = recreate_node_identity(&ctx->rewriter, node);
        } else {
            //Regular if/then/else case. Control flow joins at the immediate post dominator.
            Nodes yield_types;
            Nodes exit_args;
            Nodes lambda_args;

            Nodes old_params;
            switch (idom->node->tag) {
            case BasicBlock_TAG:
                old_params = idom->node->payload.basic_block.params;
                break;
            case AnonLambda_TAG:
                old_params = idom->node->payload.anon_lam.params;
                break;
            default:
                assert(false);
            }

            if (old_params.count == 0) {
                yield_types = empty(arena);
                exit_args = empty(arena);
                lambda_args = empty(arena);
            } else {
                const Node* types[old_params.count];
                const Node* inner_args[old_params.count];
                const Node* outer_args[old_params.count];

                for (size_t j = 0; j < old_params.count; j++) {
                    //TODO: Is this correct?
                    assert(old_params.nodes[j]->tag == Variable_TAG);
                    const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->payload.var.type);
                    //const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->type);

                    //This should always contain a qualified type?
                    //if (contains_qualified_type(types[j]))
                    types[j] = get_unqualified_type(qualified_type);

                    inner_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                    outer_args[j] = var(arena, NULL, old_params.nodes[j]->payload.var.name);
                }

                yield_types = nodes(arena, old_params.count, types);
                exit_args = nodes(arena, old_params.count, inner_args);
                lambda_args = nodes(arena, old_params.count, outer_args);
            }

            const Node* join_token = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                        .yield_types = yield_types
                        }), true), "jp");

            Node* pre_join = basic_block(arena, fn, exit_args, "exit");
            pre_join->payload.basic_block.body = join(arena, (Join) {
                    .join_point = join_token,
                    .args = exit_args
                    });

            const Node* cached = search_processed(rewriter, idom->node);
            if (cached)
                remove_dict(const Node*, rewriter->processed, idom->node);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }

            register_processed(rewriter, idom->node, pre_join);

            const Node* inner_terminator = recreate_node_identity(rewriter, node);

            remove_dict(const Node*, rewriter->processed, idom->node);
            if (cached)
                register_processed(rewriter, idom->node, cached);

            const Node* control_inner = lambda(rewriter->dst_module, singleton(join_token), inner_terminator);
            const Node* new_target = control (arena, (Control) {
                    .inside = control_inner,
                    .yield_types = yield_types
                    });

            const Node* recreated_join = rewrite_node(rewriter, idom->node);

            switch (idom->node->tag) {
            case BasicBlock_TAG: {
                const Node* outer_terminator = jump(arena, (Jump) {
                        .target = recreated_join,
                        .args = lambda_args
                        });

                const Node* anon_lam = lambda(rewriter->dst_module, lambda_args, outer_terminator);
                const Node* empty_let = let(arena, new_target, anon_lam);

                return empty_let;
            }
            case AnonLambda_TAG:
                return let(arena, new_target, recreated_join);
            default:
                assert(false);
            }
        }
    } else if (is_abstraction(node)) {
        if (!ctx->fwd_scope) {
            ctx->fwd_scope = new_scope(ctx->current_fn);
            ctx->back_scope = new_scope_flipped(ctx->current_fn);
            ctx->current_looptree = build_loop_tree(ctx->fwd_scope);
        }

        bool enters_loop = false;
        const Node* loop_entry = NULL;

        CFNode* current_node = scope_lookup(ctx->fwd_scope, node);
        LTNode* lt_node = looptree_lookup(ctx->current_looptree, node);

        assert(lt_node);
        assert(current_node);

        for (size_t i = 0; i < entries_count_list(current_node->succ_edges); i++) {
            CFEdge edge = read_list(CFEdge, current_node->succ_edges)[i];
            LTNode* lt_target = looptree_lookup(ctx->current_looptree, edge.dst->node);

            if (lt_target->parent != lt_node->parent) {
                if(lt_target->parent->depth == lt_node->parent->depth + 1) {
                    enters_loop = true;
                    loop_entry = edge.dst->node;
                }
            }
        }

        if (enters_loop) {
            CFNode* loop_entry_node = scope_lookup(ctx->fwd_scope, loop_entry);

            CFNode* exiting_node = NULL;

            for (size_t i = loop_entry_node->rpo_index + 1; i < ctx->fwd_scope->size; i++) {
                CFNode* node = ctx->fwd_scope->rpo[i];

                for (size_t j = 0; j < entries_count_list(node->succ_edges); j++) {
                    CFNode* dst = read_list(CFEdge, node->succ_edges)[j].dst;
                    LTNode* check_node = looptree_lookup(ctx->current_looptree, dst->node);

                    if (check_node->parent == lt_node->parent) {
                        exiting_node = dst;
                        break;
                    }
                }

                if (exiting_node)
                    break;
            }

            assert(exiting_node);



            Nodes yield_types;
            Nodes exit_args;
            Nodes lambda_args;

            Nodes old_params;
            switch (exiting_node->node->tag) {
            case BasicBlock_TAG:
                old_params = exiting_node->node->payload.basic_block.params;
                break;
            case AnonLambda_TAG:
                old_params = exiting_node->node->payload.anon_lam.params;
                break;
            default:
                assert(false);
            }

            if (old_params.count == 0) {
                yield_types = empty(arena);
                exit_args = empty(arena);
                lambda_args = empty(arena);
            } else {
                assert(false && "TODO");
            }

            Node* fn = NULL;

            if (is_function(node)) {
                fn = recreate_decl_header_identity(rewriter, node);
            } else {
                fn = (Node*) find_processed(rewriter, ctx->current_fn);
            }

            const Node* join_token = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                        .yield_types = yield_types
                        }), true), "jp");

            Node* pre_join = basic_block(arena, fn, exit_args, "exit");
            pre_join->payload.basic_block.body = join(arena, (Join) {
                    .join_point = join_token,
                    .args = exit_args
                    });

            const Node* cached = search_processed(rewriter, exiting_node->node);
            if (cached)
                remove_dict(const Node*, rewriter->processed, exiting_node->node);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }
            register_processed(rewriter, exiting_node->node, pre_join);

            const Node* new_node;

            if (is_function(node)) {
                recreate_decl_body_identity(rewriter, node, fn);
                new_node = fn;
                assert(false);
            }  else {
                new_node = recreate_node_identity(rewriter, node);
            }

            remove_dict(const Node*, rewriter->processed, exiting_node->node);
            if (cached)
                register_processed(rewriter, exiting_node->node, cached);

            const Node* inner_terminator = jump(arena, (Jump) {
                    .target = new_node,
                    .args = lambda_args
                    });

            const Node* control_inner = lambda(rewriter->dst_module, singleton(join_token), inner_terminator);
            const Node* new_target = control (arena, (Control) {
                    .inside = control_inner,
                    .yield_types = yield_types
                    });

            const Node* recreated_exit = rewrite_node(rewriter, exiting_node->node);

            const Node* outer_terminator = jump(arena, (Jump) {
                    .target = recreated_exit,
                    .args = lambda_args
                    });

            const Node* anon_lam = lambda(rewriter->dst_module, lambda_args, outer_terminator);
            const Node* empty_let = let(arena, new_target, anon_lam);

            Node* loop_cont = basic_block(arena, fn, exit_args, "loop");
            loop_cont->payload.basic_block.body = empty_let;

            result = loop_cont;

            assert(is_abstraction(result));

            //assert(false && "EOF");
        } else {
            result = recreate_node_identity(&ctx->rewriter, node);
        }
    } else {
        result = recreate_node_identity(&ctx->rewriter, node);
    }

    assert(result);

    ctx->current_abstraction = old_abstraction;
    return result;
}

void reconvergence_heuristics(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
        .current_fn = NULL,
        .fwd_scope = NULL,
        .back_scope = NULL,
        .current_looptree = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
