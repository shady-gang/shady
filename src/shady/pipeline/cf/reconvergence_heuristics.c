#include "shady/pass.h"

#include "ir_private.h"
#include "analysis/cfg.h"
#include "analysis/looptree.h"

#include "list.h"
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
    LTNode* lt_node = shd_loop_tree_lookup(lt, block);
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
    Context abs_ctx = *ctx;
    ctx = &abs_ctx;
    ctx->current_abstraction = node;
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    CFNode* current_node = shd_cfg_lookup(ctx->fwd_cfg, node);
    LTNode* lt_node = shd_loop_tree_lookup(ctx->current_looptree, node);
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
            shd_debugv_print("Node %s exits the loop headed at %s\n", shd_get_node_name_safe(shd_read_list(CFNode * , exiting_nodes)[i]->node), shd_get_node_name_safe(node));
        }

        size_t exiting_nodes_count = shd_list_count(exiting_nodes);
        if (exiting_nodes_count > 0) {
            Nodes nparams = shd_recreate_params(r, get_abstraction_params(node));
            Node* loop_container = basic_block_helper(a, nparams);
            shd_rewrite_annotations(r, node, loop_container);
            BodyBuilder* outer_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_container));
            Nodes inner_yield_types = shd_strip_qualifiers(a, shd_get_param_types(a, nparams));

            LARRAY(Exit, exits, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                Nodes exit_param_types = shd_rewrite_nodes(r, shd_get_param_types(ctx->rewriter.src_arena, get_abstraction_params(exiting_node->node)));

                ExitValue* exit_params = shd_arena_alloc(ctx->arena, sizeof(ExitValue) * exit_param_types.count);
                for (size_t j = 0; j < exit_param_types.count; j++) {
                    exit_params[j].alloca = shd_bld_stack_alloc(outer_bb, shd_get_unqualified_type(exit_param_types.nodes[j]));
                    exit_params[j].uniform = shd_is_qualified_type_uniform(exit_param_types.nodes[j]);
                }
                exits[i] = (Exit) {
                    .params = exit_params,
                    .params_count = exit_param_types.count,
                };
            }

            const Node* exit_destination_alloca = NULL;
            if (exiting_nodes_count > 1)
                exit_destination_alloca = shd_bld_stack_alloc(outer_bb, shd_int32_type(a));

            const Node* join_token_exit = param_helper(a, shd_as_qualified_type(join_point_type(a, (JoinPointType) {
                    .yield_types = shd_empty(a)
            }), true));
            shd_set_debug_name(join_token_exit, "jp_exit");

            const Node* join_token_continue = param_helper(a,
                                                    shd_as_qualified_type(join_point_type(a, (JoinPointType) {
                                                            .yield_types = inner_yield_types
                                                    }), true));
            shd_set_debug_name(join_token_continue, "jp_continue");

            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = shd_recreate_params(&ctx->rewriter, get_abstraction_params(exiting_node->node));

                Node* wrapper = basic_block_helper(a, exit_wrapper_params);
                shd_set_debug_name(wrapper, shd_format_string_arena(a->arena, "exit_wrapper_%d", i));
                exits[i].wrapper = wrapper;
            }

            Nodes continue_wrapper_params = shd_recreate_params(r, get_abstraction_params(node));
            Node* continue_wrapper = basic_block_helper(a, continue_wrapper_params);
            shd_set_debug_name(continue_wrapper, "continue");
            const Node* continue_wrapper_body = join(a, (Join) {
                .join_point = join_token_continue,
                .args = continue_wrapper_params,
                .mem = shd_get_abstraction_mem(continue_wrapper),
            });
            shd_set_abstraction_body(continue_wrapper, continue_wrapper_body);

            // inside the loop we want certain things to be rewritten directly!
            Context loop_ctx = *ctx;
            loop_ctx.rewriter = shd_create_children_rewriter(r);
            // replace the exit nodes by the exit wrappers
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                shd_register_processed(&loop_ctx.rewriter, exiting_node->node, exits[i].wrapper);
            }
            // ditto for the loop entry and the continue wrapper
            shd_register_processed(&loop_ctx.rewriter, node, continue_wrapper);

            Nodes inner_loop_params = shd_recreate_params(&loop_ctx.rewriter, get_abstraction_params(node));
            shd_register_processed_list(&loop_ctx.rewriter, get_abstraction_params(node), inner_loop_params);
            Node* inner_control_case = basic_block_helper(a, shd_singleton(join_token_continue));
            shd_register_processed(&loop_ctx.rewriter, shd_get_abstraction_mem(node), shd_get_abstraction_mem(inner_control_case));
            const Node* loop_body = shd_rewrite_node(&loop_ctx.rewriter, get_abstraction_body(node));

            // save the context
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];
                assert(exiting_node->node && exiting_node->node->tag != Function_TAG);
                Nodes exit_wrapper_params = get_abstraction_params(exits[i].wrapper);
                BodyBuilder* exit_wrapper_bb = shd_bld_begin(a, shd_get_abstraction_mem(exits[i].wrapper));

                for (size_t j = 0; j < exits[i].params_count; j++)
                    shd_bld_store(exit_wrapper_bb, exits[i].params[j].alloca, exit_wrapper_params.nodes[j]);
                // Set the destination if there's more than one option
                if (exiting_nodes_count > 1)
                    shd_bld_store(exit_wrapper_bb, exit_destination_alloca, shd_int32_literal(a, i));

                shd_set_abstraction_body(exits[i].wrapper, shd_bld_join(exit_wrapper_bb, join_token_exit, shd_empty(a)));
            }

            shd_set_abstraction_body(inner_control_case, loop_body);

            shd_destroy_rewriter(&loop_ctx.rewriter);

            Node* loop_outer = basic_block_helper(a, inner_loop_params);
            shd_set_debug_name(loop_outer, "loop_outer");
            BodyBuilder* inner_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_outer));
            Nodes inner_control_results = shd_bld_control(inner_bb, inner_yield_types, inner_control_case);
            // make sure what was uniform still is
            for (size_t j = 0; j < inner_control_results.count; j++) {
                if (shd_is_qualified_type_uniform(nparams.nodes[j]->type))
                    inner_control_results = shd_change_node_at_index(a, inner_control_results, j, prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(inner_control_results.nodes[j])));
            }
            shd_set_abstraction_body(loop_outer, shd_bld_jump(inner_bb, loop_outer, inner_control_results));
            Node* outer_control_case = basic_block_helper(a, shd_singleton(join_token_exit));
            shd_set_abstraction_body(outer_control_case, jump(a, (Jump) {
                .target = loop_outer,
                .args = nparams,
                .mem = shd_get_abstraction_mem(outer_control_case),
            }));
            shd_bld_control(outer_bb, shd_empty(a), outer_control_case);

            LARRAY(const Node*, exit_numbers, exiting_nodes_count);
            LARRAY(const Node*, exit_jumps, exiting_nodes_count);
            for (size_t i = 0; i < exiting_nodes_count; i++) {
                CFNode* exiting_node = shd_read_list(CFNode*, exiting_nodes)[i];

                Node* exit_bb = basic_block_helper(a, shd_empty(a));
                shd_set_debug_name(exit_bb, shd_format_string_arena(a->arena, "exit_recover_values_%s", shd_get_node_name_safe(exiting_node->node)));
                BodyBuilder* exit_recover_bb = shd_bld_begin(a, shd_get_abstraction_mem(exit_bb));

                const Node* recreated_exit = shd_rewrite_node(r, exiting_node->node);

                LARRAY(const Node*, recovered_args, exits[i].params_count);
                for (size_t j = 0; j < exits[i].params_count; j++) {
                    recovered_args[j] = shd_bld_load(exit_recover_bb, exits[i].params[j].alloca);
                    if (exits[i].params[j].uniform)
                        recovered_args[j] = prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(recovered_args[j]));
                }

                exit_numbers[i] = shd_int32_literal(a, i);
                shd_set_abstraction_body(exit_bb, shd_bld_jump(exit_recover_bb, recreated_exit, shd_nodes(a, exits[i].params_count, recovered_args)));
                exit_jumps[i] = jump_helper(a, shd_bld_mem(outer_bb), exit_bb, shd_empty(a));
            }

            const Node* outer_body;
            if (exiting_nodes_count == 1)
                outer_body = shd_bld_finish(outer_bb, exit_jumps[0]);
            else {
                const Node* loaded_destination = shd_bld_load(outer_bb, exit_destination_alloca);
                outer_body = shd_bld_finish(outer_bb, br_switch(a, (Switch) {
                    .switch_value = loaded_destination,
                    .default_jump = exit_jumps[0],
                    .case_values = shd_nodes(a, exiting_nodes_count, exit_numbers),
                    .case_jumps = shd_nodes(a, exiting_nodes_count, exit_jumps),
                    .mem = shd_bld_mem(outer_bb)
                }));
            }
            shd_set_abstraction_body(loop_container, outer_body);
            shd_destroy_list(exiting_nodes);
            return loop_container;
        }

        shd_destroy_list(exiting_nodes);
    }

    return shd_recreate_node(&ctx->rewriter, node);
}

static const Node* process_node(Context* ctx, const Node* node) {
    assert(node);

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    switch (node->tag) {
        case Function_TAG: {
            Context fn_ctx = *ctx;
            ctx = &fn_ctx;
            ctx->current_fn = NULL;
            if (!(shd_lookup_annotation(node, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;

            ctx->current_fn = node;
            ctx->fwd_cfg = build_fn_cfg(ctx->current_fn);
            ctx->rev_cfg = build_fn_cfg_flipped(ctx->current_fn);
            ctx->current_looptree = shd_new_loop_tree(ctx->fwd_cfg);

            const Node* new = process_abstraction(ctx, node);;

            shd_destroy_cfg(ctx->fwd_cfg);
            shd_destroy_cfg(ctx->rev_cfg);
            shd_destroy_loop_tree(ctx->current_looptree);
            return new;
        }
        case BasicBlock_TAG:
            if (!ctx->current_fn || !(shd_lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;
            return process_abstraction(ctx, node);
        case Branch_TAG: {
            Branch payload = node->payload.branch;
            if (!ctx->current_fn || !(shd_lookup_annotation(ctx->current_fn, "Restructure") || ctx->config->input_cf.restructure_with_heuristics))
                break;
            assert(ctx->fwd_cfg);

            CFNode* cfnode = shd_cfg_lookup(ctx->rev_cfg, ctx->current_abstraction);
            const Node* post_dominator = NULL;

            LTNode* current_loop = shd_loop_tree_lookup(ctx->current_looptree, ctx->current_abstraction)->parent;
            assert(current_loop);

            if (shd_list_count(current_loop->cf_nodes)) {
                bool leaves_loop = false;
                CFNode* current_node = shd_cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);
                for (size_t i = 0; i < shd_list_count(current_node->succ_edges); i++) {
                    CFEdge edge = shd_read_list(CFEdge, current_node->succ_edges)[i];
                    LTNode* lt_target = shd_loop_tree_lookup(ctx->current_looptree, edge.dst->node);

                    if (lt_target->parent != current_loop) {
                        leaves_loop = true;
                        break;
                    }
                }

                if (!leaves_loop) {
                    const Node* current_loop_head = shd_read_list(CFNode*, current_loop->cf_nodes)[0]->node;
                    CFG* loop_cfg = shd_new_cfg(ctx->current_fn, current_loop_head, (CFGBuildConfig) {
                        .include_structured_tails = true,
                        .lt = ctx->current_looptree,
                        .flipped = true
                    });
                    CFNode* idom_cf = shd_cfg_lookup(loop_cfg, ctx->current_abstraction)->idom;
                    if (idom_cf)
                        post_dominator = idom_cf->node;
                    shd_destroy_cfg(loop_cfg);
                }
            } else {
                post_dominator = cfnode->idom->node;
            }

            if (!post_dominator) {
                break;
            }

            if (shd_cfg_lookup(ctx->fwd_cfg, post_dominator)->idom->node != ctx->current_abstraction)
                break;

            assert(is_abstraction(post_dominator) && post_dominator->tag != Function_TAG);

            LTNode* lt_node = shd_loop_tree_lookup(ctx->current_looptree, ctx->current_abstraction);
            LTNode* idom_lt_node = shd_loop_tree_lookup(ctx->current_looptree, post_dominator);
            CFNode* current_node = shd_cfg_lookup(ctx->fwd_cfg, ctx->current_abstraction);

            assert(lt_node);
            assert(idom_lt_node);
            assert(current_node);

            //Regular if/then/else case. Control flow joins at the immediate post dominator.
            Nodes yield_types;
            Nodes exit_args;

            Nodes old_params = get_abstraction_params(post_dominator);
            LARRAY(bool, uniform_param, old_params.count);

            if (old_params.count == 0) {
                yield_types = shd_empty(a);
                exit_args = shd_empty(a);
            } else {
                LARRAY(const Node*, types, old_params.count);
                LARRAY(const Node*, inner_args,old_params.count);

                for (size_t j = 0; j < old_params.count; j++) {
                    //TODO: Is this correct?
                    assert(old_params.nodes[j]->tag == Param_TAG);
                    const Node* qualified_type = shd_rewrite_node(r, old_params.nodes[j]->payload.param.type);
                    //const Node* qualified_type = rewrite_node(rewriter, old_params.nodes[j]->type);

                    //This should always contain a qualified type?
                    //if (contains_qualified_type(types[j]))
                    types[j] = shd_get_unqualified_type(qualified_type);
                    uniform_param[j] = shd_is_qualified_type_uniform(qualified_type);
                    inner_args[j] = param_helper(a, qualified_type);
                    shd_rewrite_annotations(r, old_params.nodes[j], inner_args[j]);
                }

                yield_types = shd_nodes(a, old_params.count, types);
                exit_args = shd_nodes(a, old_params.count, inner_args);
            }

            const Node* join_token = param_helper(a, shd_as_qualified_type(join_point_type(a, (JoinPointType) {
                    .yield_types = yield_types
            }), true));
            shd_set_debug_name(join_token, "jp_postdom");

            Node* pre_join = basic_block_helper(a, exit_args);
            shd_set_debug_name(pre_join, shd_format_string_arena(a->arena, "merge_%s_%s", shd_get_node_name_safe(ctx->current_abstraction), shd_get_node_name_safe(post_dominator)));
            shd_set_abstraction_body(pre_join, join(a, (Join) {
                .join_point = join_token,
                .args = exit_args,
                .mem = shd_get_abstraction_mem(pre_join),
            }));

            Context control_ctx = *ctx;
            control_ctx.rewriter = shd_create_children_rewriter(r);
            for (size_t i = 0; i < old_params.count; i++) {
                assert(!shd_search_processed(&control_ctx.rewriter, old_params.nodes[i]));
            }

            shd_register_processed(&control_ctx.rewriter, post_dominator, pre_join);

            Node* control_case = basic_block_helper(a, shd_singleton(join_token));
            const Node* inner_terminator = branch(a, (Branch) {
                .mem = shd_get_abstraction_mem(control_case),
                .condition = shd_rewrite_node(r, payload.condition),
                .true_jump = jump_helper(a, shd_get_abstraction_mem(control_case),
                                         shd_rewrite_node(&control_ctx.rewriter, payload.true_jump->payload.jump.target),
                                         shd_rewrite_nodes(r, payload.true_jump->payload.jump.args)),
                .false_jump = jump_helper(a, shd_get_abstraction_mem(control_case),
                                          shd_rewrite_node(&control_ctx.rewriter, payload.false_jump->payload.jump.target),
                                          shd_rewrite_nodes(r, payload.false_jump->payload.jump.args)),
            });
            shd_set_abstraction_body(control_case, inner_terminator);

            shd_destroy_rewriter(&control_ctx.rewriter);

            const Node* join_target = shd_rewrite_node(r, post_dominator);

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, node->payload.branch.mem));
            Nodes results = shd_bld_control(bb, yield_types, control_case);
            // make sure what was uniform still is
            for (size_t j = 0; j < old_params.count; j++) {
                if (uniform_param[j])
                    results = shd_change_node_at_index(a, results, j, prim_op_helper(a, subgroup_assume_uniform_op, shd_empty(a), shd_singleton(results.nodes[j])));
            }
            return shd_bld_jump(bb, join_target, results);
        }
        default: break;
    }
    return shd_recreate_node(r, node);
}

Module* shd_pass_reconvergence_heuristics(const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    aconfig.optimisations.inline_single_use_bbs = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));

    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .config = config,
        .current_fn = NULL,
        .fwd_cfg = NULL,
        .rev_cfg = NULL,
        .current_looptree = NULL,
        .arena = shd_new_arena(),
    };

    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_arena(ctx.arena);
    return dst;
}
