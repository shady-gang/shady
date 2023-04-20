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

static bool in_loop(LoopTree* lt, const Node* entry, const Node* block) {
    LTNode* lt_node = looptree_lookup(lt, block);
    assert(lt_node);
    LTNode* parent = lt_node->parent;
    assert(parent);

    while (parent) {
        if (entries_count_list(parent->cf_nodes) != 1)
            return false;

        if (read_list(CFNode*, parent->cf_nodes)[0]->node == entry)
            return true;

        parent = parent->parent;
    }

    return false;
}

//TODO: This is massively inefficient.
static void gather_exiting_nodes(LoopTree* lt, const CFNode* entry, const CFNode* block, struct List* exiting_nodes) {
    if (!in_loop(lt, entry->node, block->node)) {
        append_list(CFNode*, exiting_nodes, block);
        return;
    }

    for (size_t i = 0; i < entries_count_list(block->dominates); i++) {
        const CFNode* target = read_list(CFNode*, block->dominates)[i];
        gather_exiting_nodes(lt, entry, target, exiting_nodes);
    }
}

static const Node* process_abstraction(Context* ctx, const Node* node) {
    assert(is_abstraction(node));
    Rewriter* rewriter = &ctx->rewriter;
    IrArena* arena = rewriter->dst_arena;

    CFNode* current_node = scope_lookup(ctx->fwd_scope, node);
    LTNode* lt_node = looptree_lookup(ctx->current_looptree, node);

    assert(current_node);
    assert(lt_node);

    bool is_loop_entry = false;
    const CFNode* loop_entry_node = NULL;
    if (entries_count_list(lt_node->parent->cf_nodes) == 1) {
        loop_entry_node = read_list(CFNode*, lt_node->parent->cf_nodes)[0];
        const Node* loop_header = loop_entry_node->node;
        is_loop_entry = loop_header == node;
    }

    if (is_loop_entry) {
        struct List * exiting_nodes = new_list(CFNode*);
        gather_exiting_nodes(ctx->current_looptree, loop_entry_node, loop_entry_node, exiting_nodes);
        const CFNode* exiting_node = NULL;
        if (entries_count_list(exiting_nodes) > 0)
            exiting_node = read_list(CFNode*, exiting_nodes)[0];
        for (size_t i = 1; i < entries_count_list(exiting_nodes); i++) {
            const CFNode* next_exiting_node = read_list(CFNode*, exiting_nodes)[i];
            const CFNode* exiting_node_back = scope_lookup(ctx->back_scope, exiting_node->node);
            const CFNode* next_exiting_node_back = scope_lookup(ctx->back_scope, next_exiting_node->node);
            const CFNode* shared_post_dominator = least_common_ancestor(exiting_node_back, next_exiting_node_back);
            exiting_node = scope_lookup(ctx->fwd_scope, shared_post_dominator->node);
        }

        destroy_list(exiting_nodes);

        if (exiting_node) {
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

            assert(!is_function(node));
            Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

            const Node* join_token_exit = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = yield_types
            }), true), "jp_exit");
            const Node* join_token_continue = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = yield_types
            }), true), "jp_continue");

            const Node* pre_join_exit;
            const Node* pre_join_exit_join = join(arena, (Join) {
                    .join_point = join_token_exit,
                    .args = exit_args
            });
            switch (exiting_node->node->tag) {
                case BasicBlock_TAG: {
                    Node* pre_join_exit_bb = basic_block(arena, fn, exit_args, "exit");
                    pre_join_exit_bb->payload.basic_block.body = pre_join_exit_join;
                    pre_join_exit = pre_join_exit_bb;
                    break;
                }
                case AnonLambda_TAG:
                    pre_join_exit = lambda(arena, exit_args, pre_join_exit_join);
                    break;
                default:
                    assert(false);
            }

            const Node* pre_join_continue;
            const Node* pre_join_continue_join = join(arena, (Join) {
                .join_point = join_token_continue,
                .args = exit_args
            });
            switch (loop_entry_node->node->tag) {
                case BasicBlock_TAG: {
                    Node* pre_join_continue_bb = basic_block(arena, fn, exit_args, "continue");
                    pre_join_continue_bb->payload.basic_block.body = pre_join_continue_join;
                    pre_join_continue = pre_join_continue_bb;
                    break;
                }
                case AnonLambda_TAG:
                    pre_join_continue = lambda(arena, exit_args, pre_join_continue_join);
                    break;
                default:
                    assert(false);
            }

            const Node* cached_exit = search_processed(rewriter, exiting_node->node);
            if (cached_exit)
                remove_dict(const Node*, is_declaration(exiting_node->node) ? rewriter->decls_map : rewriter->map, exiting_node->node);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }
            register_processed(rewriter, exiting_node->node, pre_join_exit);

            const Node* cached_entry = search_processed(rewriter, loop_entry_node->node);
            if (cached_entry)
                remove_dict(const Node*, is_declaration(loop_entry_node->node) ? rewriter->decls_map : rewriter->map, loop_entry_node->node);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }
            register_processed(rewriter, loop_entry_node->node, pre_join_continue);

            const Node* new_terminator;
            switch (node->tag) {
                case BasicBlock_TAG:
                    new_terminator = rewrite_node(rewriter, node->payload.basic_block.body);
                    break;
                case AnonLambda_TAG:
                    new_terminator = rewrite_node(rewriter, node->payload.anon_lam.body);
                    break;
                default:
                    assert(false);
            }
            Node* loop_inner = basic_block(arena, fn, exit_args, "loop_inner");
            loop_inner->payload.basic_block.body = new_terminator;

            remove_dict(const Node*, is_declaration(exiting_node->node) ? rewriter->decls_map : rewriter->map, exiting_node->node);
            if (cached_exit)
                register_processed(rewriter, exiting_node->node, cached_exit);

            remove_dict(const Node*, is_declaration(loop_entry_node->node) ? rewriter->decls_map : rewriter->map, loop_entry_node->node);
            if (cached_entry)
                register_processed(rewriter, loop_entry_node->node, cached_entry);

            const Node* inner_terminator = jump(arena, (Jump) {
                .target = loop_inner,
                .args = lambda_args
            });

            const Node* control_inner_lambda = lambda(arena, singleton(join_token_continue), inner_terminator);
            const Node* inner_control = control (arena, (Control) {
                .inside = control_inner_lambda,
                .yield_types = yield_types
            });

            Node* loop_outer = basic_block(arena, fn, exit_args, "loop_outer");
            const Node* loop_terminator = jump(arena, (Jump) {
                .target = loop_outer,
                .args = lambda_args
            });

            const Node* anon_lam = lambda(arena, lambda_args, loop_terminator);
            const Node* inner_control_let = let(arena, inner_control, anon_lam);

            loop_outer->payload.basic_block.body = inner_control_let;

            const Node* control_outer_lambda = lambda(arena, singleton(join_token_exit), loop_terminator);
            const Node* outer_control = control (arena, (Control) {
                .inside = control_outer_lambda,
                .yield_types = yield_types
            });

            const Node* recreated_exit = rewrite_node(rewriter, exiting_node->node);
            const Node* outer_terminator = jump(arena, (Jump) {
                .target = recreated_exit,
                .args = lambda_args
            });

            const Node* anon_lam_exit = lambda(arena, lambda_args, outer_terminator);
            const Node* outer_control_let = let(arena, outer_control, anon_lam_exit);

            const Node* loop_container;
            switch (node->tag) {
                case BasicBlock_TAG: {
                    Node* bb = basic_block(arena, fn, exit_args, node->payload.basic_block.name);
                    bb->payload.basic_block.body = outer_control_let;
                    loop_container = bb;
                    break;
                }
                case AnonLambda_TAG:
                    loop_container = lambda(arena, exit_args, outer_control_let);
                    break;
                default:
                    assert(false);
            }
            return loop_container;
        }
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    assert(node);

    Rewriter* rewriter = &ctx->rewriter;
    IrArena* arena = rewriter->dst_arena;

    if (!ctx->current_fn || !lookup_annotation(ctx->current_fn, "Restructure")) {
        return recreate_node_identity(&ctx->rewriter, node);
    }

    switch (node->tag) {
        case Function_TAG: {
            Context new_context = *ctx;
            ctx = &new_context;
            ctx->current_fn = node;
            ctx->fwd_scope = new_scope(ctx->current_fn);
            ctx->back_scope = new_scope_flipped(ctx->current_fn);
            ctx->current_looptree = build_loop_tree(ctx->fwd_scope);

            const Node* new = process_abstraction(ctx, node);;

            destroy_scope(ctx->fwd_scope);
            destroy_scope(ctx->back_scope);
            destroy_loop_tree(ctx->current_looptree);
            return new;
        }
        case AnonLambda_TAG:
        case BasicBlock_TAG:
            return process_abstraction(ctx, node);
        case Branch_TAG: {
            assert(ctx->fwd_scope);

            CFNode* cfnode = scope_lookup(ctx->back_scope, ctx->current_abstraction);

            CFNode* idom = NULL;

            LTNode* current_loop = looptree_lookup(ctx->current_looptree, ctx->current_abstraction)->parent;
            assert(current_loop);

            if (entries_count_list(current_loop->cf_nodes)) {
                bool leaves_loop = false;
                CFNode* current_node = scope_lookup(ctx->fwd_scope, ctx->current_abstraction);
                for (size_t i = 0; i < entries_count_list(current_node->succ_edges); i++) {
                    CFEdge edge = read_list(CFEdge, current_node->succ_edges)[i];
                    LTNode* lt_target = looptree_lookup(ctx->current_looptree, edge.dst->node);

                    if (lt_target->parent != current_loop) {
                        leaves_loop = true;
                        break;
                    }
                }

                if (!leaves_loop) {
                    const Node* current_loop_head = read_list(CFNode*, current_loop->cf_nodes)[0]->node;
                    Scope* loop_scope = new_scope_lt_flipped(current_loop_head, ctx->current_looptree);
                    idom = scope_lookup(loop_scope, ctx->current_abstraction)->idom;
                    destroy_scope(loop_scope);
                }
            } else {
                idom = cfnode->idom;
            }

            if(!idom || !idom->node) {
                break;
            }

            LTNode* lt_node = looptree_lookup(ctx->current_looptree, ctx->current_abstraction);
            LTNode* idom_lt_node = looptree_lookup(ctx->current_looptree, idom->node);
            CFNode* current_node = scope_lookup(ctx->fwd_scope, ctx->current_abstraction);

            assert(lt_node);
            assert(idom_lt_node);
            assert(current_node);

            Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

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
                remove_dict(const Node*, is_declaration(idom->node) ? rewriter->decls_map : rewriter->map, idom->node);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }

            register_processed(rewriter, idom->node, pre_join);

            const Node* inner_terminator = recreate_node_identity(rewriter, node);

            remove_dict(const Node*, is_declaration(idom->node) ? rewriter->decls_map : rewriter->map, idom->node);
            if (cached)
                register_processed(rewriter, idom->node, cached);

            const Node* control_inner = lambda(arena, singleton(join_token), inner_terminator);
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

                    const Node* anon_lam = lambda(arena, lambda_args, outer_terminator);
                    const Node* empty_let = let(arena, new_target, anon_lam);

                    return empty_let;
                }
                case AnonLambda_TAG:
                    return let(arena, new_target, recreated_join);
                default:
                    assert(false);
            }
        }
        default: break;
    }
    return recreate_node_identity(rewriter, node);
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
