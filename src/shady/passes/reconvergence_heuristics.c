#include "shady/pass.h"

#include "../type.h"
#include "../ir_private.h"
#include "../transform/ir_gen_helpers.h"

#include "../analysis/cfg.h"
#include "../analysis/looptree.h"

#include "list.h"
#include "dict.h"
#include "log.h"
#include "portability.h"
#include "util.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    const CompilerConfig* config;
    Arena* arena;
    const Node* current_fn;
    const Node* current_abstraction;
    CFG* fwd_cfg;
    CFG* rev_cfg;
    LoopTree* current_looptree;
} Context;

static bool in_loop(LoopTree* lt, const Node* entry, const Node* block) {
    LTNode* lt_node = looptree_lookup(lt, block);
    assert(lt_node);
    LTNode* parent = lt_node->parent;
    assert(parent);

    while (parent) {
        if (shd_list_count(parent->cf_nodes) != 1)
            return false;

        if (shd_read_list(CFNode*, parent->cf_nodes)[0]->node == entry)
            return true;

        parent = parent->parent;
    }

    return false;
}

//TODO: This is massively inefficient.
static void gather_exiting_nodes(LoopTree* lt, const CFNode* entry, const CFNode* block, struct List* exiting_nodes) {
    if (!in_loop(lt, entry->node, block->node)) {
        shd_list_append(CFNode*, exiting_nodes, block);
        return;
    }

    for (size_t i = 0; i < shd_list_count(block->dominates); i++) {
        const CFNode* target = shd_read_list(CFNode*, block->dominates)[i];
        gather_exiting_nodes(lt, entry, target, exiting_nodes);
    }
}

static void find_unbound_vars(const Node* exiting_node, struct Dict* bound_set, struct Dict* free_set, struct List* leaking) {
    const Node* v;
    size_t i = 0;
    while (shd_dict_iter(free_set, &i, &v, NULL)) {
        if (shd_dict_find_key(const Node*, bound_set, v))
            continue;

        log_string(DEBUGVV, "Found variable used outside it's control scope: ");
        log_node(DEBUGVV, v);
        log_string(DEBUGVV, " (exiting_node:");
        log_node(DEBUGVV, exiting_node);
        log_string(DEBUGVV, " )\n");

        shd_list_append(const Node*, leaking, v);
    }
}

typedef struct {
    const Node* alloca;
    bool uniform;
} ExitValue;

typedef struct {
    ExitValue* params;
    size_t params_count;

    Node* wrapper;
} Exit;

static const Node* process_abstraction(Context* ctx, const Node* node) {
    assert(node && is_abstraction(node));
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
        if (shd_list_count(lt_node->parent->cf_nodes) == 1)
            if (shd_read_list(CFNode*, lt_node->parent->cf_nodes)[0]->node == node) {
                loop_header = lt_node->parent;
                assert(loop_header->type == LF_HEAD);
                assert(shd_list_count(loop_header->cf_nodes) == 1 && "only reducible loops are handled");
                is_loop_entry = true;
            }
    }

    if (is_loop_entry) {
        assert(!is_function(node));

        struct List* exiting_nodes = shd_new_list(CFNode*);
        gather_exiting_nodes(ctx->current_looptree, current_node, current_node, exiting_nodes);

        for (size_t i = 0; i < shd_list_count(exiting_nodes); i++) {
            debugv_print("Node %s exits the loop headed at %s\n", get_abstraction_name_safe(shd_read_list(CFNode *, exiting_nodes)[i]->node), get_abstraction_name_safe(node));
        }

        size_t exiting_nodes_count = shd_list_count(exiting_nodes);
        if (exiting_nodes_count > 0) {
            Nodes nparams = recreate_params(rewriter, get_abstraction_params(node));
            Node* loop_container = basic_block(arena, nparams, node->payload.basic_block.name);
            BodyBuilder* outer_bb = begin_body_with_mem(arena, get_abstraction_mem(loop_container));
            Nodes inner_yield_types = strip_qualifiers(arena, get_param_types(arena, nparams));

            LARRAY(Exit, exits, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                Nodes exit_param_types = rewrite_nodes(rewriter, get_param_types(ctx->rewriter.src_arena, get_abstraction_params(exiting_node->node)));

                ExitValue* exit_params = shd_arena_alloc(ctx->arena, sizeof(ExitValue) * exit_param_types.count);
                for (size_t j = 0; j < exit_param_types.count; j++) {
                    exit_params[j].alloca = gen_stack_alloc(outer_bb, get_unqualified_type(exit_param_types.nodes[j]));
                    exit_params[j].uniform = is_qualified_type_uniform(exit_param_types.nodes[j]);
                }
                exits[i] = (Exit) {
                    .params = exit_params,
                    .params_count = exit_param_types.count,
                };
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

            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = recreate_params(&ctx->rewriter, get_abstraction_params(exiting_node->node));

                Node* wrapper = basic_block(arena, exit_wrapper_params, format_string_arena(arena->arena, "exit_wrapper_%d", i));
                exits[i].wrapper = wrapper;
            }

            Nodes continue_wrapper_params = recreate_params(rewriter, get_abstraction_params(node));
            Node* continue_wrapper = basic_block(arena, continue_wrapper_params, "continue");
            const Node* continue_wrapper_body = join(arena, (Join) {
                .join_point = join_token_continue,
                .args = continue_wrapper_params,
                .mem = get_abstraction_mem(continue_wrapper),
            });
            set_abstraction_body(continue_wrapper, continue_wrapper_body);

            // replace the exit nodes by the exit wrappers
            LARRAY(const Node**, cached_exits, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                cached_exits[i] = search_processed(rewriter, exiting_node->node);
                if (cached_exits[i])
                    shd_dict_remove(const Node*, rewriter->map, exiting_node->node);
                register_processed(rewriter, exiting_node->node, exits[i].wrapper);
            }
            // ditto for the loop entry and the continue wrapper
            const Node** cached_entry = search_processed(rewriter, node);
            if (cached_entry)
                shd_dict_remove(const Node*, rewriter->map, node);
            register_processed(rewriter, node, continue_wrapper);

            // make sure we haven't started rewriting this...
            // for (size_t i = 0; i < old_params.count; i++) {
            //     assert(!search_processed(rewriter, old_params.nodes[i]));
            // }

            struct Dict* old_map = rewriter->map;
            rewriter->map = shd_clone_dict(rewriter->map);
            Nodes inner_loop_params = recreate_params(rewriter, get_abstraction_params(node));
            register_processed_list(rewriter, get_abstraction_params(node), inner_loop_params);
            Node* inner_control_case = case_(arena, singleton(join_token_continue));
            register_processed(rewriter, get_abstraction_mem(node), get_abstraction_mem(inner_control_case));
            const Node* loop_body = rewrite_node(rewriter, get_abstraction_body(node));

            // save the context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = get_abstraction_params(exits[i].wrapper);
                BodyBuilder* exit_wrapper_bb = begin_body_with_mem(arena, get_abstraction_mem(exits[i].wrapper));

                for (size_t j = 0; j < exits[i].params_count; j++)
                    gen_store(exit_wrapper_bb, exits[i].params[j].alloca, exit_wrapper_params.nodes[j]);
                // Set the destination if there's more than one option
                if (exiting_nodes_count > 1)
                    gen_store(exit_wrapper_bb, exit_destination_alloca, int32_literal(arena, i));

                set_abstraction_body(exits[i].wrapper, finish_body_with_join(exit_wrapper_bb, join_token_exit, empty(arena)));
            }

            set_abstraction_body(inner_control_case, loop_body);

            shd_destroy_dict(rewriter->map);
            rewriter->map = old_map;
            //register_processed_list(rewriter, get_abstraction_params(node), nparams);

            // restore the old context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                shd_dict_remove(const Node*, rewriter->map, shd_read_list(CFNode *, exiting_nodes)[i]->node);
                if (cached_exits[i])
                    register_processed(rewriter, shd_read_list(CFNode*, exiting_nodes)[i]->node, *cached_exits[i]);
            }
            shd_dict_remove(const Node*, rewriter->map, node);
            if (cached_entry)
                register_processed(rewriter, node, *cached_entry);

            Node* loop_outer = basic_block(arena, inner_loop_params, "loop_outer");
            BodyBuilder* inner_bb = begin_body_with_mem(arena, get_abstraction_mem(loop_outer));
            Nodes inner_control_results = gen_control(inner_bb, inner_yield_types, inner_control_case);
            // make sure what was uniform still is
            for (size_t j = 0; j < inner_control_results.count; j++) {
                if (is_qualified_type_uniform(nparams.nodes[j]->type))
                    inner_control_results = change_node_at_index(arena, inner_control_results, j, prim_op_helper(arena, subgroup_assume_uniform_op, empty(arena), singleton(inner_control_results.nodes[j])));
            }
            set_abstraction_body(loop_outer, finish_body_with_jump(inner_bb, loop_outer, inner_control_results));
            Node* outer_control_case = case_(arena, singleton(join_token_exit));
            set_abstraction_body(outer_control_case, jump(arena, (Jump) {
                .target = loop_outer,
                .args = nparams,
                .mem = get_abstraction_mem(outer_control_case),
            }));
            gen_control(outer_bb, empty(arena), outer_control_case);

            LARRAY(const Node*, exit_numbers, exiting_nodes_count);
            LARRAY(const Node*, exit_jumps, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];

                Node* exit_bb = basic_block(arena, empty(arena), format_string_arena(arena->arena, "exit_recover_values_%s", get_abstraction_name_safe(exiting_node->node)));
                BodyBuilder* exit_recover_bb = begin_body_with_mem(arena, get_abstraction_mem(exit_bb));

                const Node* recreated_exit = rewrite_node(rewriter, exiting_node->node);

                LARRAY(const Node*, recovered_args, exits[i].params_count);
                for (size_t j = 0; j < exits[i].params_count; j++) {
                    recovered_args[j] = gen_load(exit_recover_bb, exits[i].params[j].alloca);
                    if (exits[i].params[j].uniform)
                        recovered_args[j] = prim_op_helper(arena, subgroup_assume_uniform_op, empty(arena), singleton(recovered_args[j]));
                }

                exit_numbers[i] = int32_literal(arena, i);
                set_abstraction_body(exit_bb, finish_body_with_jump(exit_recover_bb, recreated_exit, nodes(arena, exits[i].params_count, recovered_args)));
                exit_jumps[i] = jump_helper(arena, exit_bb, empty(arena), bb_mem(outer_bb));
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
            shd_destroy_list(exiting_nodes);
            return loop_container;
        }

        shd_destroy_list(exiting_nodes);
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
            ctx->current_fn = NULL;
            if (!(lookup_annotation(node, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;

            ctx->current_fn = node;
            ctx->fwd_cfg = build_fn_cfg(ctx->current_fn);
            ctx->rev_cfg = build_fn_cfg_flipped(ctx->current_fn);
            ctx->current_looptree = build_loop_tree(ctx->fwd_cfg);

            const Node* new = process_abstraction(ctx, node);;

            destroy_cfg(ctx->fwd_cfg);
            destroy_cfg(ctx->rev_cfg);
            destroy_loop_tree(ctx->current_looptree);
            return new;
        }
        case Constant_TAG: {
            ctx = &new_context;
            ctx->current_fn = NULL;
            r = &ctx->rewriter;
            break;
        }
        case BasicBlock_TAG:
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;
            return process_abstraction(ctx, node);
        case Branch_TAG: {
            Branch payload = node->payload.branch;
            if (!ctx->current_fn || !(lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;
            assert(ctx->fwd_cfg);

            CFNode* cfnode = cfg_lookup(ctx->rev_cfg, ctx->current_abstraction);
            const Node* post_dominator = NULL;

            LTNode* current_loop = looptree_lookup(ctx->current_looptree, ctx->current_abstraction)->parent;
            assert(current_loop);

            if (shd_list_count(current_loop->cf_nodes)) {
                bool leaves_loop = false;
                CFNode* current_node = cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);
                for (size_t i = 0; i < shd_list_count(current_node->succ_edges); i++) {
                    CFEdge edge = shd_read_list(CFEdge, current_node->succ_edges)[i];
                    LTNode* lt_target = looptree_lookup(ctx->current_looptree, edge.dst->node);

                    if (lt_target->parent != current_loop) {
                        leaves_loop = true;
                        break;
                    }
                }

                if (!leaves_loop) {
                    const Node* current_loop_head = shd_read_list(CFNode*, current_loop->cf_nodes)[0]->node;
                    CFG* loop_cfg = build_cfg(ctx->current_fn, current_loop_head, (CFGBuildConfig) {
                        .include_structured_tails = true,
                        .lt = ctx->current_looptree,
                        .flipped = true
                    });
                    CFNode* idom_cf = cfg_lookup(loop_cfg, ctx->current_abstraction)->idom;
                    if (idom_cf)
                        post_dominator = idom_cf->node;
                    destroy_cfg(loop_cfg);
                }
            } else {
                post_dominator = cfnode->idom->node;
            }

            if (!post_dominator) {
                break;
            }

            if (cfg_lookup(ctx->fwd_cfg, post_dominator)->idom->node != ctx->current_abstraction)
                break;

            assert(is_abstraction(post_dominator) && post_dominator->tag != Function_TAG);

            LTNode* lt_node = looptree_lookup(ctx->current_looptree, ctx->current_abstraction);
            LTNode* idom_lt_node = looptree_lookup(ctx->current_looptree, post_dominator);
            CFNode* current_node = cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);

            assert(lt_node);
            assert(idom_lt_node);
            assert(current_node);

            Node* fn = (Node*) find_processed(r, ctx->current_fn);

            //Regular if/then/else case. Control flow joins at the immediate post dominator.
            Nodes yield_types;
            Nodes exit_args;

            Nodes old_params = get_abstraction_params(post_dominator);
            LARRAY(bool, uniform_param, old_params.count);

            if (old_params.count == 0) {
                yield_types = empty(a);
                exit_args = empty(a);
            } else {
                LARRAY(const Node*, types, old_params.count);
                LARRAY(const Node*, inner_args,old_params.count);

                for (size_t j = 0; j < old_params.count; j++) {
                    //TODO: Is this correct?
                    assert(old_params.nodes[j]->tag == Param_TAG);
                    const Node* qualified_type = rewrite_node(r, old_params.nodes[j]->payload.param.type);
                    //const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->type);

                    //This should always contain a qualified type?
                    //if (contains_qualified_type(types[j]))
                    types[j] = get_unqualified_type(qualified_type);
                    uniform_param[j] = is_qualified_type_uniform(qualified_type);
                    inner_args[j] = param(a, qualified_type, old_params.nodes[j]->payload.param.name);
                }

                yield_types = nodes(a, old_params.count, types);
                exit_args = nodes(a, old_params.count, inner_args);
            }

            const Node* join_token = param(a, qualified_type_helper(join_point_type(a, (JoinPointType) {
                    .yield_types = yield_types
            }), true), "jp_postdom");

            Node* pre_join = basic_block(a, exit_args, format_string_arena(a->arena, "merge_%s_%s", get_abstraction_name_safe(ctx->current_abstraction) , get_abstraction_name_safe(post_dominator)));
            set_abstraction_body(pre_join, join(a, (Join) {
                .join_point = join_token,
                .args = exit_args,
                .mem = get_abstraction_mem(pre_join),
            }));

            const Node** cached = search_processed(r, post_dominator);
            if (cached)
                shd_dict_remove(const Node*, is_declaration(post_dominator) ? r->decls_map : r->map, post_dominator);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!search_processed(r, old_params.nodes[i]));
            }

            register_processed(r, post_dominator, pre_join);

            Node* control_case = case_(a, singleton(join_token));
            const Node* inner_terminator = branch(a, (Branch) {
                .mem = get_abstraction_mem(control_case),
                .condition = rewrite_node(r, payload.condition),
                .true_jump = jump_helper(a, rewrite_node(r, payload.true_jump->payload.jump.target), rewrite_nodes(r, payload.true_jump->payload.jump.args), get_abstraction_mem(control_case)),
                .false_jump = jump_helper(a, rewrite_node(r, payload.false_jump->payload.jump.target), rewrite_nodes(r, payload.false_jump->payload.jump.args), get_abstraction_mem(control_case)),
            });
            set_abstraction_body(control_case, inner_terminator);

            shd_dict_remove(const Node*, is_declaration(post_dominator) ? r->decls_map : r->map, post_dominator);
            if (cached)
                register_processed(r, post_dominator, *cached);

            const Node* join_target = rewrite_node(r, post_dominator);

            BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, node->payload.branch.mem));
            Nodes results = gen_control(bb, yield_types, control_case);
            // make sure what was uniform still is
            for (size_t j = 0; j < old_params.count; j++) {
                if (uniform_param[j])
                    results = change_node_at_index(a, results, j, prim_op_helper(a, subgroup_assume_uniform_op, empty(a), singleton(results.nodes[j])));
            }
            return finish_body_with_jump(bb, join_target, results);
        }
        default: break;
    }
    return recreate_node_identity(r, node);
}

Module* reconvergence_heuristics(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    aconfig.optimisations.inline_single_use_bbs = true;
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));

    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .current_fn = NULL,
        .fwd_cfg = NULL,
        .rev_cfg = NULL,
        .current_looptree = NULL,
        .arena = shd_new_arena(),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    shd_destroy_arena(ctx.arena);
    return dst;
}
