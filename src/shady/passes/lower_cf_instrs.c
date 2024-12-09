#include "shady/pass.h"

#include "../analysis/cfg.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

#include <assert.h>

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    Node* current_fn;

    struct Dict* structured_join_tokens;
    CFG* cfg;
} Context;

static const Node* process_node(Context* ctx, const Node* node) {
    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    Context sub_ctx = *ctx;
    if (node->tag == Function_TAG) {
        Node* fun = shd_recreate_node_head(&ctx->rewriter, node);
        sub_ctx.disable_lowering = shd_lookup_annotation(fun, "Structured");
        sub_ctx.current_fn = fun;
        sub_ctx.cfg = build_fn_cfg(node);
        shd_set_abstraction_body(fun, shd_rewrite_node(&sub_ctx.rewriter, node->payload.fun.body));
        shd_destroy_cfg(sub_ctx.cfg);
        return fun;
    } else if (node->tag == Constant_TAG) {
        sub_ctx.cfg = NULL;
        sub_ctx.current_fn = NULL;
        ctx = &sub_ctx;
    }

    if (ctx->disable_lowering)
        return shd_recreate_node(&ctx->rewriter, node);

    switch (node->tag) {
        case If_TAG: {
            If payload = node->payload.if_instr;
            bool has_false_branch = payload.if_false;
            Nodes yield_types = shd_rewrite_nodes(&ctx->rewriter, node->payload.if_instr.yield_types);
            const Node* nmem = shd_rewrite_node(r, node->payload.if_instr.mem);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = false,
            });
            const Node* jp = param(a, jp_type, "if_join");
            Nodes jps = shd_singleton(jp);
            shd_dict_insert(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            const Node* true_block = shd_rewrite_node(r, payload.if_true);

            const Node* false_block;
            if (has_false_branch) {
                false_block = shd_rewrite_node(r, payload.if_false);
            } else {
                assert(yield_types.count == 0);
                false_block = basic_block(a, shd_nodes(a, 0, NULL), shd_make_unique_name(a, "if_false"));
                shd_set_abstraction_body((Node*) false_block, join(a, (Join) { .join_point = jp, .args = shd_nodes(a, 0, NULL), .mem = shd_get_abstraction_mem(false_block) }));
            }

            Node* control_case = basic_block(a, shd_singleton(jp), NULL);
            const Node* control_body = branch(a, (Branch) {
                .condition = shd_rewrite_node(r, node->payload.if_instr.condition),
                .true_jump = jump_helper(a, shd_get_abstraction_mem(control_case), true_block, shd_empty(a)),
                .false_jump = jump_helper(a, shd_get_abstraction_mem(control_case), false_block, shd_empty(a)),
                .mem = shd_get_abstraction_mem(control_case),
            });
            shd_set_abstraction_body(control_case, control_body);

            BodyBuilder* bb = shd_bld_begin(a, nmem);
            Nodes results = shd_bld_control(bb, yield_types, control_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), shd_rewrite_node(r, payload.tail), results));
        }
        // TODO: match
        case Loop_TAG: {
            Loop payload = node->payload.loop_instr;
            const Node* old_loop_block = payload.body;

            Nodes yield_types = shd_rewrite_nodes(&ctx->rewriter, node->payload.loop_instr.yield_types);
            Nodes param_types = shd_rewrite_nodes(&ctx->rewriter, shd_get_param_types(a, get_abstraction_params(old_loop_block)));
            param_types = shd_strip_qualifiers(a, param_types);

            const Type* break_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = false,
            });
            const Type* continue_jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = param_types }),
                .is_uniform = false,
            });
            const Node* break_point = param(a, break_jp_type, "loop_break_point");
            const Node* continue_point = param(a, continue_jp_type, "loop_continue_point");
            Nodes jps = mk_nodes(a, break_point, continue_point);
            shd_dict_insert(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            Nodes new_params = shd_recreate_params(&ctx->rewriter, get_abstraction_params(old_loop_block));
            Node* loop_header_block = basic_block(a, new_params, shd_make_unique_name(a, "loop_header"));

            BodyBuilder* inner_bb = shd_bld_begin(a, shd_get_abstraction_mem(loop_header_block));
            Node* inner_control_case = case_(a, shd_singleton(continue_point));
            shd_set_abstraction_body(inner_control_case, jump_helper(a, shd_get_abstraction_mem(inner_control_case),
                                                                     shd_rewrite_node(r, old_loop_block), new_params));
            Nodes args = shd_bld_control(inner_bb, param_types, inner_control_case);

            shd_set_abstraction_body(loop_header_block, shd_bld_finish(inner_bb, jump(a, (Jump) { .target = loop_header_block, .args = args, .mem = shd_bld_mem(inner_bb) })));

            Node* outer_control_case = case_(a, shd_singleton(break_point));
            const Node* first_iteration_jump = jump(a, (Jump) {
                .target = loop_header_block,
                .args = shd_rewrite_nodes(r, payload.initial_args),
                .mem = shd_get_abstraction_mem(outer_control_case),
            });
            shd_set_abstraction_body(outer_control_case, first_iteration_jump);

            BodyBuilder* bb = shd_bld_begin(a, shd_rewrite_node(r, payload.mem));
            Nodes results = shd_bld_control(bb, yield_types, outer_control_case);
            return shd_bld_finish(bb, jump_helper(a, shd_bld_mem(bb), shd_rewrite_node(r, payload.tail), results));
        }
        case MergeSelection_TAG: {
            MergeSelection payload = node->payload.merge_selection;
            const Node* root_mem = shd_get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = shd_cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                if(body->tag == If_TAG || body->tag == Match_TAG) {
                    selection_instr = body;
                    break;
                }
                dom = dom->idom;
            }

            if (!selection_instr) {
                shd_error_print("Scoping error: Failed to find a dominating selection construct for ");
                shd_log_node(ERROR, node);
                shd_error_print(".\n");
                shd_error_die();
            }

            Nodes* jps = shd_dict_find_value(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 1);
            const Node* jp = shd_first(*jps);
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem
            });
        }
        case MergeContinue_TAG: {
            MergeContinue payload = node->payload.merge_continue;
            const Node* root_mem = shd_get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = shd_cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
            CFNode* dom = cfnode->idom;
            const Node* loop_start = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                if (body->tag == Loop_TAG) {
                    loop_start = body;
                    break;
                }
                dom = dom->idom;
            }

            if (!loop_start) {
                shd_error_print("Scoping error: Failed to find a dominating loop construct for ");
                shd_log_node(ERROR, node);
                shd_error_print(".\n");
                shd_error_die();
            }

            Nodes* jps = shd_dict_find_value(const Node*, Nodes, ctx->structured_join_tokens, loop_start);
            assert(jps && jps->count == 2);
            const Node* jp = jps->nodes[1];
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem,
            });
        }
        case MergeBreak_TAG: {
            MergeBreak payload = node->payload.merge_break;
            const Node* root_mem = shd_get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = shd_cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
            CFNode* dom = cfnode->idom;
            const Node* loop_start = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                if (body->tag == Loop_TAG) {
                    loop_start = body;
                    break;
                }
                dom = dom->idom;
            }

            if (!loop_start) {
                shd_error_print("Scoping error: Failed to find a dominating loop construct for ");
                shd_log_node(ERROR, node);
                shd_error_print(".\n");
                shd_error_die();
            }

            Nodes* jps = shd_dict_find_value(const Node*, Nodes, ctx->structured_join_tokens, loop_start);
            assert(jps && jps->count == 2);
            const Node* jp = shd_first(*jps);
            assert(jp);
            const Node* nmem = shd_rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = shd_rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem,
            });
        }
        default: break;
    }
    return shd_recreate_node(&ctx->rewriter, node);
}

KeyHash shd_hash_node(const Node**);
bool shd_compare_node(const Node**, const Node**);

Module* shd_pass_lower_cf_instrs(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *shd_get_arena_config(shd_module_get_arena(src));
    IrArena* a = shd_new_ir_arena(&aconfig);
    Module* dst = shd_new_module(a, shd_module_get_name(src));
    Context ctx = {
        .rewriter = shd_create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .structured_join_tokens = shd_new_dict(const Node*, Nodes, (HashFn) shd_hash_node, (CmpFn) shd_compare_node),
    };
    shd_rewrite_module(&ctx.rewriter);
    shd_destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.structured_join_tokens);
    return dst;
}

