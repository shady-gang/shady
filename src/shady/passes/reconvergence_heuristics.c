#include "pass.h"

#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "../analysis/cfg.h"
#include "../analysis/looptree.h"
#include "../analysis/free_variables.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    const Node* current_fn;
    const Node* current_abstraction;
    CFG* fwd_cfg;
    CFG* rev_cfg;
    LoopTree* current_looptree;
    struct Dict* live_vars;
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

static void find_unbound_vars(const Node* exiting_node, struct Dict* bound_set, struct Dict* free_set, struct List* leaking) {
    const Node* v;
    size_t i = 0;
    while (dict_iter(free_set, &i, &v, NULL)) {
        if (find_key_dict(const Node*, bound_set, v))
            continue;

        log_string(DEBUGVV, "Found variable used outside it's control scope: ");
        log_node(DEBUGVV, v);
        log_string(DEBUGVV, " (exiting_node:");
        log_node(DEBUGVV, exiting_node);
        log_string(DEBUGVV, " )\n");

        append_list(const Node*, leaking, v);
    }
}

static const Node* process_abstraction(Context* ctx, const Node* node) {
    assert(is_abstraction(node));
    Context new_context = *ctx;
    ctx = &new_context;
    ctx->current_abstraction = node;
    Rewriter* rewriter = &ctx->rewriter;
    IrArena* arena = rewriter->dst_arena;

    CFNode* current_node = cfg_lookup(ctx->fwd_cfg, node);
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
            debugv_print("Node %s exits the loop headed at %s\n", get_abstraction_name_safe(read_list(CFNode*, exiting_nodes)[i]->node), get_abstraction_name_safe(node));
        }

        size_t exiting_nodes_count = entries_count_list(exiting_nodes);
        if (exiting_nodes_count > 0) {
            Nodes nparams = recreate_params(rewriter, get_abstraction_params(node));
            Node* loop_container = basic_block(arena, nparams, node->payload.basic_block.name);
            BodyBuilder* outer_bb = begin_body_with_mem(arena, get_abstraction_mem(loop_container));
            Nodes inner_yield_types = strip_qualifiers(arena, get_param_types(arena, nparams));

            CFNode* cf_pre = cfg_lookup(ctx->fwd_cfg, node);
            // assert(cf_pre->idom && "cfg entry nodes can't be loop headers anyhow");
            // cf_pre = cf_pre->idom;
            CFNodeVariables* pre = *find_value_dict(CFNode*, CFNodeVariables*, ctx->live_vars, cf_pre);

            LARRAY(Nodes, exit_param_allocas, exiting_nodes_count);
            LARRAY(struct List*, leaking, exiting_nodes_count);
            LARRAY(Nodes, exit_fwd_allocas, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                Nodes exit_param_types = rewrite_nodes(rewriter, get_param_types(ctx->rewriter.src_arena, get_abstraction_params(exiting_node->node)));
                LARRAY(const Node*, exit_param_allocas_tmp, exit_param_types.count);
                for (size_t j = 0; j < exit_param_types.count; j++)
                    exit_param_allocas_tmp[j] = gen_stack_alloc(outer_bb, get_unqualified_type(exit_param_types.nodes[j]));
                exit_param_allocas[i] = nodes(arena, exit_param_types.count, exit_param_allocas_tmp);

                // Search for what's required after the exit but not in scope at the loop header
                // this is similar to the LCSSA constraint, but here it's because controls have hard scopes
                CFNode* cf_post = cfg_lookup(ctx->fwd_cfg, exiting_node->node);
                CFNodeVariables* post = *find_value_dict(CFNode*, CFNodeVariables*, ctx->live_vars, cf_post);
                leaking[i] = new_list(const Type*);
                find_unbound_vars(exiting_node->node, pre->bound_by_dominators_set, post->free_set, leaking[i]);

                size_t leaking_count = entries_count_list(leaking[i]);
                LARRAY(const Node*, exit_fwd_allocas_tmp, leaking_count);
                for (size_t j = 0; j < leaking_count; j++)
                    exit_fwd_allocas_tmp[j] = gen_stack_alloc(outer_bb, rewrite_node(rewriter, get_unqualified_type(read_list(const Node*, leaking[i])[j]->type)));
                exit_fwd_allocas[i] = nodes(arena, leaking_count, exit_fwd_allocas_tmp);
            }

            const Node* exit_destination_alloca = NULL;
            if (exiting_nodes_count > 1)
                exit_destination_alloca = gen_stack_alloc(outer_bb, int32_type(arena));

            const Node* join_token_exit = param(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                .yield_types = empty(arena)
            }), true), "jp_exit");

            const Node* join_token_continue = param(arena, qualified_type_helper(join_point_type(arena, (JoinPointType) {
                .yield_types = inner_yield_types
            }), true), "jp_continue");

            LARRAY(Node*, exit_wrappers, exiting_nodes_count);
            LARRAY(Node*, exit_helpers, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                exit_helpers[i] = basic_block(arena, empty(arena), format_string_arena(arena->arena, "exit_helper_%d", i));

                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = recreate_params(&ctx->rewriter, get_abstraction_params(exiting_node->node));

                switch (exiting_node->node->tag) {
                    case BasicBlock_TAG: {
                        Node* pre_join_exit_bb = basic_block(arena, exit_wrapper_params, format_string_arena(arena->arena, "exit_wrapper_%d", i));
                        set_abstraction_body(pre_join_exit_bb, jump_helper(arena, exit_helpers[i], empty(arena), get_abstraction_mem(pre_join_exit_bb)));
                        exit_wrappers[i] = pre_join_exit_bb;
                        break;
                    }
                    default:
                        assert(false);
                }
            }

            Nodes continue_wrapper_params = recreate_params(rewriter, get_abstraction_params(node));
            const Node* continue_wrapper_body = join(arena, (Join) {
                .join_point = join_token_continue,
                .args = continue_wrapper_params
            });
            Node* continue_wrapper;
            switch (node->tag) {
                case BasicBlock_TAG: {
                    continue_wrapper = basic_block(arena, continue_wrapper_params, "continue");
                    set_abstraction_body(continue_wrapper, continue_wrapper_body);
                    break;
                }
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
            Nodes inner_loop_params = recreate_params(rewriter, get_abstraction_params(node));
            register_processed_list(rewriter, get_abstraction_params(node), inner_loop_params);
            const Node* loop_body = recreate_node_identity(rewriter, get_abstraction_body(node));

            // save the context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = get_abstraction_params(exit_wrappers[i]);
                BodyBuilder* exit_wrapper_bb = begin_body_with_mem(arena, get_abstraction_mem(exit_helpers[i]));

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

                set_abstraction_body(exit_helpers[i], exit_wrapper_body);
            }

            destroy_dict(rewriter->map);
            rewriter->map = old_map;
            //register_processed_list(rewriter, get_abstraction_params(node), nparams);

            // restore the old context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                remove_dict(const Node*, rewriter->map, read_list(CFNode*, exiting_nodes)[i]->node);
                if (cached_exits[i])
                    register_processed(rewriter, read_list(CFNode*, exiting_nodes)[i]->node, cached_exits[i]);
            }
            remove_dict(const Node*, rewriter->map, node);
            if (cached_entry)
                register_processed(rewriter, node, cached_entry);

            Node* loop_outer = basic_block(arena, inner_loop_params, "loop_outer");
            BodyBuilder* inner_bb = begin_body_with_mem(arena, get_abstraction_mem(loop_outer));
            Node* inner_control_case = case_(arena, singleton(join_token_continue));
            set_abstraction_body(inner_control_case, loop_body);
            Nodes inner_control_results = gen_control(inner_bb, inner_yield_types, inner_control_case);

            set_abstraction_body(loop_outer, finish_body(inner_bb, jump(arena, (Jump) {
                .target = loop_outer,
                .args = inner_control_results
            })));
            Node* outer_control_case = case_(arena, singleton(join_token_exit));
            set_abstraction_body(outer_control_case, jump(arena, (Jump) {
                .target = loop_outer,
                .args = nparams
            }));
            gen_control(outer_bb, empty(arena), outer_control_case);

            LARRAY(const Node*, exit_numbers, exiting_nodes_count);
            LARRAY(const Node*, exit_jumps, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = read_list(CFNode*, exiting_nodes)[i];

                Node* exit_bb = basic_block(arena, empty(arena), format_string_arena(arena->arena, "exit_recover_values_%s", get_abstraction_name_safe(exiting_node->node)));
                BodyBuilder* exit_recover_bb = begin_body_with_mem(arena, get_abstraction_mem(exit_bb));

                // recover the context
                for (size_t j = 0; j < exit_fwd_allocas[i].count; j++) {
                    const Node* recovered = gen_load(exit_recover_bb, exit_fwd_allocas[i].nodes[j]);
                    register_processed(rewriter, read_list(const Node*, leaking[i])[j], recovered);
                }

                const Node* recreated_exit = rewrite_node(rewriter, exiting_node->node);

                LARRAY(const Node*, recovered_args, exit_param_allocas[i].count);
                for (size_t j = 0; j < exit_param_allocas[i].count; j++)
                    recovered_args[j] = gen_load(exit_recover_bb, exit_param_allocas[i].nodes[j]);

                exit_numbers[i] = int32_literal(arena, i);
                if (recreated_exit->tag == BasicBlock_TAG) {
                    set_abstraction_body(exit_bb, finish_body(exit_recover_bb, jump(arena, (Jump) {
                        .target = recreated_exit,
                        .args = nodes(arena, exit_param_allocas[i].count, recovered_args),
                    })));
                } else {
                    // TODO: rewrite
                    assert(get_abstraction_params(recreated_exit).count == 0);
                    error("")
                    // assert(recreated_exit->tag == Case_TAG);
                    // exit_bb->payload.basic_block.body = finish_body(exit_recover_bb, let(arena, quote_helper(arena, nodes(arena, exit_param_allocas[i].count, recovered_args)), recreated_exit));
                }
                exit_jumps[i] = jump_helper(arena, exit_bb, empty(arena), bb_mem(outer_bb));
                destroy_list(leaking[i]);
            }

            const Node* outer_body;
            if (exiting_nodes_count == 1)
                outer_body = finish_body(outer_bb, exit_jumps[0]);
            else {
                const Node* loaded_destination = gen_load(outer_bb, exit_destination_alloca);
                outer_body = finish_body(outer_bb, br_switch(arena, (Switch) {
                    .switch_value = loaded_destination,
                    .default_jump = exit_jumps[0],
                    .case_values = nodes(arena, exiting_nodes_count, exit_numbers),
                    .case_jumps = nodes(arena, exiting_nodes_count, exit_jumps),
                    .mem = bb_mem(outer_bb)
                }));
            }
            set_abstraction_body(loop_container, outer_body);
            destroy_list(exiting_nodes);
            return loop_container;
        }

        destroy_list(exiting_nodes);
    }

    return recreate_node_identity(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    assert(node);

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    Context new_context = *ctx;

    switch (node->tag) {
        case Function_TAG: {
            ctx = &new_context;
            ctx->current_fn = node;
            ctx->fwd_cfg = build_cfg(ctx->current_fn, ctx->current_fn, NULL, false);
            ctx->rev_cfg = build_cfg(ctx->current_fn, ctx->current_fn, NULL, true);
            ctx->current_looptree = build_loop_tree(ctx->fwd_cfg);
            ctx->live_vars = compute_cfg_variables_map(ctx->fwd_cfg, CfgVariablesAnalysisFlagDomBoundSet | CfgVariablesAnalysisFlagLiveSet | CfgVariablesAnalysisFlagFreeSet);

            const Node* new = process_abstraction(ctx, node);;

            destroy_cfg(ctx->fwd_cfg);
            destroy_cfg(ctx->rev_cfg);
            destroy_loop_tree(ctx->current_looptree);
            destroy_cfg_variables_map(ctx->live_vars);
            return new;
        }
        case Constant_TAG: {
            ctx = &new_context;
            ctx->current_fn = NULL;
            r = &ctx->rewriter;
            break;
        }
        case BasicBlock_TAG:
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->hacks.restructure_everything))
                break;
            return process_abstraction(ctx, node);
        case Branch_TAG: {
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->hacks.restructure_everything))
                break;
            assert(ctx->fwd_cfg);

            CFNode* cfnode = cfg_lookup(ctx->rev_cfg, ctx->current_abstraction);
            const Node* idom = NULL;

            LTNode* current_loop = looptree_lookup(ctx->current_looptree, ctx->current_abstraction)->parent;
            assert(current_loop);

            if (entries_count_list(current_loop->cf_nodes)) {
                bool leaves_loop = false;
                CFNode* current_node = cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);
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
                    CFG* loop_cfg = build_cfg(ctx->current_fn, current_loop_head, ctx->current_looptree, true);
                    CFNode* idom_cf = cfg_lookup(loop_cfg, ctx->current_abstraction)->idom;
                    if (idom_cf)
                        idom = idom_cf->node;
                    destroy_cfg(loop_cfg);
                }
            } else {
                idom = cfnode->idom->node;
            }

            if (!idom) {
                break;
            }

            if (cfg_lookup(ctx->fwd_cfg, idom)->idom->node != ctx->current_abstraction)
                break;

            assert(is_abstraction(idom) && idom->tag != Function_TAG);

            LTNode* lt_node = looptree_lookup(ctx->current_looptree, ctx->current_abstraction);
            LTNode* idom_lt_node = looptree_lookup(ctx->current_looptree, idom);
            CFNode* current_node = cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);

            assert(lt_node);
            assert(idom_lt_node);
            assert(current_node);

            Node* fn = (Node*) find_processed(r, ctx->current_fn);

            //Regular if/then/else case. Control flow joins at the immediate post dominator.
            Nodes yield_types;
            Nodes exit_args;

            Nodes old_params = get_abstraction_params(idom);

            if (old_params.count == 0) {
                yield_types = empty(a);
                exit_args = empty(a);
            } else {
                LARRAY(const Node*, types,old_params.count);
                LARRAY(const Node*, inner_args,old_params.count);

                for (size_t j = 0; j < old_params.count; j++) {
                    //TODO: Is this correct?
                    assert(old_params.nodes[j]->tag == Param_TAG);
                    const Node* qualified_type = rewrite_node(r, old_params.nodes[j]->payload.param.type);
                    //const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->type);

                    //This should always contain a qualified type?
                    //if (contains_qualified_type(types[j]))
                    types[j] = get_unqualified_type(qualified_type);

                    inner_args[j] = param(a, qualified_type, old_params.nodes[j]->payload.param.name);
                }

                yield_types = nodes(a, old_params.count, types);
                exit_args = nodes(a, old_params.count, inner_args);
            }

            const Node* join_token = param(a, qualified_type_helper(join_point_type(a, (JoinPointType) {
                    .yield_types = yield_types
            }), true), "jp_postdom");

            Node* pre_join = basic_block(a, exit_args, format_string_arena(a->arena, "merge_%s_%s", get_abstraction_name_safe(ctx->current_abstraction) , get_abstraction_name_safe(idom)));
            set_abstraction_body(pre_join, join(a, (Join) {
                .join_point = join_token,
                .args = exit_args
            }));

            const Node* cached = search_processed(r, idom);
            if (cached)
                remove_dict(const Node*, is_declaration(idom) ? r->decls_map : r->map, idom);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(r, old_params.nodes[i]));
            }

            register_processed(r, idom, pre_join);

            const Node* inner_terminator = recreate_node_identity(r, node);

            remove_dict(const Node*, is_declaration(idom) ? r->decls_map : r->map, idom);
            if (cached)
                register_processed(r, idom, cached);

            Node* control_case = case_(a, singleton(join_token));
            set_abstraction_body(control_case, inner_terminator);
            const Node* join_target = rewrite_node(r, idom);

            switch (idom->tag) {
                case BasicBlock_TAG: {
                    BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, node->payload.branch.mem));
                    Nodes results = gen_control(bb, yield_types, control_case);
                    return finish_body(bb, jump(a, (Jump) {
                        .target = join_target,
                        .args = results
                    }));
                }
                default:
                    assert(false);
            }
        }
        default: break;
    }
    return recreate_node_identity(r, node);
}

Module* reconvergence_heuristics(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .current_fn = NULL,
        .fwd_cfg = NULL,
        .rev_cfg = NULL,
        .current_looptree = NULL,
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    return dst;
}
