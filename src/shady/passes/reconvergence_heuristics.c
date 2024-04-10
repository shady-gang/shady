#include "shady/ir.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "util.h"

#include "../type.h"
#include "../rewrite.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "../analysis/scope.h"
#include "../analysis/looptree.h"
#include "../analysis/free_variables.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    const Node* current_fn;
    const Node* current_abstraction;
    Scope* fwd_scope;
    Scope* back_scope;
    LoopTree* current_looptree;
    struct Dict* scope_vars;
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

static void find_unbound_vars(const Node* node, struct Dict* bound_set, struct Dict* free_set, struct List* leaking) {
    const Node* free_post;
    size_t i = 0;
    while (dict_iter(free_set, &i, &free_post, NULL)) {
        const Node* bound_pre;
        size_t j = 0;
        while (dict_iter(bound_set, &j, &bound_pre, NULL)) {
            if (bound_pre == free_post) {
                goto next;
            }
        }

        log_string(DEBUGVV, "Found variable used outside it's control scope: ");
        log_node(DEBUGVV, free_post);
        log_string(DEBUGVV, " (original:");
        log_node(DEBUGVV, node);
        log_string(DEBUGVV, " )\n");

        append_list(const Node*, leaking, free_post);

        next:;
    }
}

static const Node* process_abstraction(Context* ctx, const Node* node) {
    assert(is_abstraction(node));
    Context new_context = *ctx;
    ctx = &new_context;
    ctx->current_abstraction = node;
    Rewriter* rewriter = &ctx->rewriter;
    IrArena* arena = rewriter->dst_arena;

    CFNode* current_node = scope_lookup(ctx->fwd_scope, node);
    LTNode* lt_node = looptree_lookup(ctx->current_looptree, node);
    LTNode* loop_header = NULL;

    assert(current_node);
    assert(lt_node);

    bool is_loop_entry = false;
    if (lt_node->parent && lt_node->parent->type == LF_HEAD) {
        if (entries_count_list(lt_node->parent->cf_nodes) == 1)
            if (read_list(CFNode*, lt_node->parent->cf_nodes)[0]->node == node) {
                loop_header = lt_node->parent;
                assert(loop_header->type == LF_HEAD);
                assert(entries_count_list(loop_header->cf_nodes) == 1 && "only reducible loops are handled");
                is_loop_entry = true;
            }
    }

    if (is_loop_entry) {
        assert(!is_function(node));

        struct List* exiting_nodes = new_list(CFNode*);
        gather_exiting_nodes(ctx->current_looptree, current_node, current_node, exiting_nodes);

        for (size_t i = 0; i < entries_count_list(exiting_nodes); i++) {
            debugv_print("Node %s exits the loop headed at %s\n", get_abstraction_name(read_list(CFNode*, exiting_nodes)[i]->node), get_abstraction_name(node));
        }

        BodyBuilder* outer_bb = begin_body(arena);

        size_t exiting_nodes_count = entries_count_list(exiting_nodes);
        if (exiting_nodes_count > 0) {
            Nodes nparams = recreate_variables(rewriter, get_abstraction_params(node));
            Nodes inner_yield_types = strip_qualifiers(arena, get_variables_types(arena, nparams));

            CFNode* cf_pre = scope_lookup(ctx->fwd_scope, node);
            // assert(cf_pre->idom && "cfg entry nodes can't be loop headers anyhow");
            // cf_pre = cf_pre->idom;
            CFNodeVariables* pre = *find_value_dict(CFNode*, CFNodeVariables*, ctx->scope_vars, cf_pre);

            LARRAY(Nodes, exit_param_allocas, exiting_nodes_count);
            LARRAY(struct List*, leaking, exiting_nodes_count);
            LARRAY(Nodes, exit_fwd_allocas, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                Nodes exit_param_types = rewrite_nodes(rewriter, get_variables_types(ctx->rewriter.src_arena, get_abstraction_params(exiting_node->node)));
                LARRAY(const Node*, exit_param_allocas_tmp, exit_param_types.count);
                for (size_t j = 0; j < exit_param_types.count; j++)
                    exit_param_allocas_tmp[j] = gen_primop_e(outer_bb, alloca_op, singleton(get_unqualified_type(exit_param_types.nodes[j])), empty(arena));
                exit_param_allocas[i] = nodes(arena, exit_param_types.count, exit_param_allocas_tmp);

                // Search for what's required after the exit but not in scope at the loop header
                // this is similar to the LCSSA constraint, but here it's because controls have hard scopes
                CFNode* cf_post = scope_lookup(ctx->fwd_scope, exiting_node->node);
                CFNodeVariables* post = *find_value_dict(CFNode*, CFNodeVariables*, ctx->scope_vars, cf_post);
                leaking[i] = new_list(const Type*);
                find_unbound_vars(exiting_node->node, pre->bound_set, post->free_set, leaking[i]);

                size_t leaking_count = entries_count_list(leaking[i]);
                LARRAY(const Node*, exit_fwd_allocas_tmp, leaking_count);
                for (size_t j = 0; j < leaking_count; j++)
                    exit_fwd_allocas_tmp[j] = gen_primop_e(outer_bb, alloca_op, singleton(rewrite_node(rewriter, get_unqualified_type(read_list(const Node*, leaking[i])[j]->type))), empty(arena));
                exit_fwd_allocas[i] = nodes(arena, leaking_count, exit_fwd_allocas_tmp);
            }

            const Node* exit_destination_alloca = NULL;
            if (exiting_nodes_count > 1)
                exit_destination_alloca = gen_primop_e(outer_bb, alloca_op, singleton(int32_type(arena)), empty(arena));

            Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

            const Node* join_token_exit = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = empty(arena)
            }), true), "jp_exit");

            const Node* join_token_continue = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = inner_yield_types
            }), true), "jp_continue");

            LARRAY(const Node*, exit_wrappers, exiting_nodes_count);
            LARRAY(Node*, exit_helpers, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                exit_helpers[i] = basic_block(arena, fn, empty(arena), format_string_arena(arena->arena, "exit_helper_%d", i));

                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = recreate_variables(&ctx->rewriter, get_abstraction_params(exiting_node->node));

                switch (exiting_node->node->tag) {
                    case BasicBlock_TAG: {
                        Node* pre_join_exit_bb = basic_block(arena, fn, exit_wrapper_params, format_string_arena(arena->arena, "exit_wrapper_%d", i));
                        pre_join_exit_bb->payload.basic_block.body = jump_helper(arena, exit_helpers[i], empty(arena));
                        exit_wrappers[i] = pre_join_exit_bb;
                        break;
                    }
                    case Case_TAG:
                        exit_wrappers[i] = case_(arena, exit_wrapper_params, jump_helper(arena, exit_helpers[i], empty(arena)));
                        break;
                    default:
                        assert(false);
                }
            }

            Nodes continue_wrapper_params = recreate_variables(rewriter, get_abstraction_params(node));
            const Node* continue_wrapper_body = join(arena, (Join) {
                .join_point = join_token_continue,
                .args = continue_wrapper_params
            });
            const Node* continue_wrapper;
            switch (node->tag) {
                case BasicBlock_TAG: {
                    Node* pre_join_continue_bb = basic_block(arena, fn, continue_wrapper_params, "continue");
                    pre_join_continue_bb->payload.basic_block.body = continue_wrapper_body;
                    continue_wrapper = pre_join_continue_bb;
                    break;
                }
                case Case_TAG:
                    continue_wrapper = case_(arena, continue_wrapper_params, continue_wrapper_body);
                    break;
                default:
                    assert(false);
            }

            // replace the exit nodes by the exit wrappers
            LARRAY(const Node*, cached_exits, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                cached_exits[i] = search_processed(rewriter, exiting_node->node);
                if (cached_exits[i])
                    remove_dict(const Node*, rewriter->map, exiting_node->node);
                register_processed(rewriter, exiting_node->node, exit_wrappers[i]);
            }
            // ditto for the loop entry and the continue wrapper
            const Node* cached_entry = search_processed(rewriter, node);
            if (cached_entry)
                remove_dict(const Node*, rewriter->map, node);
            register_processed(rewriter, node, continue_wrapper);

            // make sure we haven't started rewriting this...
            // for (size_t i = 0; i < old_params.count; i++) {
            //     assert(!search_processed(rewriter, old_params.nodes[i]));
            // }

            struct Dict* old_map = rewriter->map;
            rewriter->map = clone_dict(rewriter->map);
            Nodes inner_loop_params = recreate_variables(rewriter, get_abstraction_params(node));
            register_processed_list(rewriter, get_abstraction_params(node), inner_loop_params);
            const Node* loop_body = recreate_node_identity(rewriter, get_abstraction_body(node));

            // save the context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = get_abstraction_params(exit_helpers[i]);
                BodyBuilder* exit_wrapper_bb = begin_body(arena);

                for (size_t j = 0; j < exit_param_allocas[i].count; j++)
                    gen_store(exit_wrapper_bb, exit_param_allocas[i].nodes[j], exit_wrapper_params.nodes[j]);
                if (exiting_nodes_count > 1)
                    gen_store(exit_wrapper_bb, exit_destination_alloca, int32_literal(arena, i));

                for (size_t j = 0; j < exit_fwd_allocas[i].count; j++) {
                    gen_store(exit_wrapper_bb, exit_fwd_allocas[i].nodes[j], rewrite_node(rewriter, read_list(const Node*, leaking[i])[j]));
                }

                const Node* exit_wrapper_body = finish_body(exit_wrapper_bb, join(arena, (Join) {
                    .join_point = join_token_exit,
                    .args = empty(arena)
                }));

                exit_helpers[i]->payload.basic_block.body = exit_wrapper_body;
            }

            destroy_dict(rewriter->map);
            rewriter->map = old_map;
            register_processed_list(rewriter, get_abstraction_params(node), nparams);

            // restore the old context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                remove_dict(const Node*, rewriter->map, read_list(CFNode*, exiting_nodes)[i]->node);
                if (cached_exits[i])
                    register_processed(rewriter, read_list(CFNode*, exiting_nodes)[i]->node, cached_exits[i]);
            }
            remove_dict(const Node*, rewriter->map, node);
            if (cached_entry)
                register_processed(rewriter, node, cached_entry);

            BodyBuilder* inner_bb = begin_body(arena);
            const Node* inner_control = control(arena, (Control) {
                .inside = case_(arena, singleton(join_token_continue), loop_body),
                .yield_types = inner_yield_types
            });
            Nodes inner_control_results = bind_instruction(inner_bb, inner_control);

            Node* loop_outer = basic_block(arena, fn, inner_loop_params, "loop_outer");

            loop_outer->payload.basic_block.body = finish_body(inner_bb, jump(arena, (Jump) {
                    .target = loop_outer,
                    .args = inner_control_results
            }));
            const Node* outer_control = control(arena, (Control) {
                .inside = case_(arena, singleton(join_token_exit), jump(arena, (Jump) {
                        .target = loop_outer,
                        .args = nparams
                })),
                .yield_types = empty(arena)
            });

            bind_instruction(outer_bb, outer_control);

            LARRAY(const Node*, exit_numbers, exiting_nodes_count);
            LARRAY(const Node*, exit_jumps, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                BodyBuilder* exit_recover_bb = begin_body(arena);

                // recover the context
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                for (size_t j = 0; j < exit_fwd_allocas[i].count; j++) {
                    const Node* recovered = gen_load(exit_recover_bb, exit_fwd_allocas[i].nodes[j]);
                    register_processed(rewriter, read_list(const Node*, leaking[i])[j], recovered);
                }

                const Node* recreated_exit = rewrite_node(rewriter, exiting_node->node);

                LARRAY(const Node*, recovered_args, exit_param_allocas[i].count);
                for (size_t j = 0; j < exit_param_allocas[i].count; j++)
                    recovered_args[j] = gen_load(exit_recover_bb, exit_param_allocas[i].nodes[j]);

                exit_numbers[i] = int32_literal(arena, i);
                Node* exit_bb = basic_block(arena, fn, empty(arena), format_string_arena(arena->arena, "exit_recover_values_%s", get_abstraction_name(exiting_node->node)));
                if (recreated_exit->tag == BasicBlock_TAG) {
                    exit_bb->payload.basic_block.body = finish_body(exit_recover_bb, jump(arena, (Jump) {
                        .target = recreated_exit,
                        .args = nodes(arena, exit_param_allocas[i].count, recovered_args),
                    }));
                } else {
                    assert(recreated_exit->tag == Case_TAG);
                    exit_bb->payload.basic_block.body = finish_body(exit_recover_bb, let(arena, quote_helper(arena, nodes(arena, exit_param_allocas[i].count, recovered_args)), recreated_exit));
                }
                exit_jumps[i] = jump_helper(arena, exit_bb, empty(arena));
                destroy_list(leaking[i]);
            }

            const Node* outer_body;
            if (exiting_nodes_count == 1)
                outer_body = finish_body(outer_bb, exit_jumps[0]->payload.jump.target->payload.basic_block.body);
            else {
                const Node* loaded_destination = gen_load(outer_bb, exit_destination_alloca);
                outer_body = finish_body(outer_bb, br_switch(arena, (Switch) {
                    .switch_value = loaded_destination,
                    .default_jump = exit_jumps[0],
                    .case_values = nodes(arena, exiting_nodes_count, exit_numbers),
                    .case_jumps = nodes(arena, exiting_nodes_count, exit_jumps),
                }));
            }

            const Node* loop_container;
            switch (node->tag) {
                case BasicBlock_TAG: {
                    Node* bb = basic_block(arena, fn, nparams, node->payload.basic_block.name);
                    bb->payload.basic_block.body = outer_body;
                    loop_container = bb;
                    break;
                }
                case Case_TAG:
                    loop_container = case_(arena, nparams, outer_body);
                    break;
                default:
                    assert(false);
            }
            destroy_list(exiting_nodes);
            return loop_container;
        }

        destroy_list(exiting_nodes);
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    assert(node);

    Rewriter* rewriter = &ctx->rewriter;
    IrArena* arena = rewriter->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Context new_context = *ctx;
            ctx = &new_context;
            ctx->current_fn = node;
            ctx->fwd_scope = new_scope(ctx->current_fn);
            ctx->back_scope = new_scope_flipped(ctx->current_fn);
            ctx->current_looptree = build_loop_tree(ctx->fwd_scope);
            ctx->scope_vars = compute_scope_variables_map(ctx->fwd_scope);

            const Node* new = process_abstraction(ctx, node);;

            destroy_scope(ctx->fwd_scope);
            destroy_scope(ctx->back_scope);
            destroy_loop_tree(ctx->current_looptree);
            destroy_scope_variables_map(ctx->scope_vars);
            return new;
        }
        case Case_TAG:
        case BasicBlock_TAG:
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->hacks.restructure_everything))
                break;
            return process_abstraction(ctx, node);
        case Branch_TAG: {
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->hacks.restructure_everything))
                break;
            assert(ctx->fwd_scope);

            CFNode* cfnode = scope_lookup(ctx->back_scope, ctx->current_abstraction);
            const Node* idom = NULL;

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
                    CFNode* idom_cf = scope_lookup(loop_scope, ctx->current_abstraction)->idom;
                    if (idom_cf)
                        idom = idom_cf->node;
                    destroy_scope(loop_scope);
                }
            } else {
                idom = cfnode->idom->node;
            }

            if (!idom) {
                break;
            }

            if (scope_lookup(ctx->fwd_scope, idom)->idom->node!= ctx->current_abstraction)
                break;

            assert(is_abstraction(idom) && idom->tag != Function_TAG);

            LTNode* lt_node = looptree_lookup(ctx->current_looptree, ctx->current_abstraction);
            LTNode* idom_lt_node = looptree_lookup(ctx->current_looptree, idom);
            CFNode* current_node = scope_lookup(ctx->fwd_scope, ctx->current_abstraction);

            assert(lt_node);
            assert(idom_lt_node);
            assert(current_node);

            Node* fn = (Node*) find_processed(rewriter, ctx->current_fn);

            //Regular if/then/else case. Control flow joins at the immediate post dominator.
            Nodes yield_types;
            Nodes exit_args;
            Nodes lambda_args;

            Nodes old_params = get_abstraction_params(idom);

            if (old_params.count == 0) {
                yield_types = empty(arena);
                exit_args = empty(arena);
                lambda_args = empty(arena);
            } else {
                LARRAY(const Node*, types,old_params.count);
                LARRAY(const Node*, inner_args,old_params.count);
                LARRAY(const Node*, outer_args,old_params.count);

                for (size_t j = 0; j < old_params.count; j++) {
                    //TODO: Is this correct?
                    assert(old_params.nodes[j]->tag == Variable_TAG);
                    const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->payload.var.type);
                    //const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->type);

                    //This should always contain a qualified type?
                    //if (contains_qualified_type(types[j]))
                    types[j] = get_unqualified_type(qualified_type);

                    inner_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                    outer_args[j] = var(arena, qualified_type, old_params.nodes[j]->payload.var.name);
                }

                yield_types = nodes(arena, old_params.count, types);
                exit_args = nodes(arena, old_params.count, inner_args);
                lambda_args = nodes(arena, old_params.count, outer_args);
            }

            const Node* join_token = var(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                    .yield_types = yield_types
            }), true), "jp_postdom");

            Node* pre_join = basic_block(arena, fn, exit_args, format_string_arena(arena->arena, "merge_%s_%s", get_abstraction_name(ctx->current_abstraction) , get_abstraction_name(idom)));
            pre_join->payload.basic_block.body = join(arena, (Join) {
                .join_point = join_token,
                .args = exit_args
            });

            const Node* cached = search_processed(rewriter, idom);
            if (cached)
                remove_dict(const Node*, is_declaration(idom) ? rewriter->decls_map : rewriter->map, idom);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(rewriter, old_params.nodes[i]));
            }

            register_processed(rewriter, idom, pre_join);

            const Node* inner_terminator = recreate_node_identity(rewriter, node);

            remove_dict(const Node*, is_declaration(idom) ? rewriter->decls_map : rewriter->map, idom);
            if (cached)
                register_processed(rewriter, idom, cached);

            const Node* control_inner = case_(arena, singleton(join_token), inner_terminator);
            const Node* new_target = control(arena, (Control) {
                .inside = control_inner,
                .yield_types = yield_types
            });

            const Node* recreated_join = rewrite_node(rewriter, idom);

            switch (idom->tag) {
                case BasicBlock_TAG: {
                    const Node* outer_terminator = jump(arena, (Jump) {
                        .target = recreated_join,
                        .args = lambda_args
                    });

                    const Node* c = case_(arena, lambda_args, outer_terminator);
                    const Node* empty_let = let(arena, new_target, c);

                    return empty_let;
                }
                case Case_TAG:
                    return let(arena, new_target, recreated_join);
                default:
                    assert(false);
            }
        }
        default: break;
    }
    return recreate_node_identity(rewriter, node);
}

Module* reconvergence_heuristics(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .current_fn = NULL,
        .fwd_scope = NULL,
        .back_scope = NULL,
        .current_looptree = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
