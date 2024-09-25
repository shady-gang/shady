#include "shady/pass.h"

#include "../type.h"
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
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        sub_ctx.disable_lowering = lookup_annotation(fun, "Structured");
        sub_ctx.current_fn = fun;
        sub_ctx.cfg = build_fn_cfg(node);
        set_abstraction_body(fun, rewrite_node(&sub_ctx.rewriter, node->payload.fun.body));
        destroy_cfg(sub_ctx.cfg);
        return fun;
    } else if (node->tag == Constant_TAG) {
        sub_ctx.cfg = NULL;
        sub_ctx.current_fn = NULL;
        ctx = &sub_ctx;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case If_TAG: {
            If payload = node->payload.if_instr;
            bool has_false_branch = payload.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.if_instr.yield_types);
            const Node* nmem = rewrite_node(r, node->payload.if_instr.mem);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = false,
            });
            const Node* jp = param(a, jp_type, "if_join");
            Nodes jps = singleton(jp);
            shd_dict_insert(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            const Node* true_block = rewrite_node(r, payload.if_true);

            const Node* false_block;
            if (has_false_branch) {
                false_block = rewrite_node(r, payload.if_false);
            } else {
                assert(yield_types.count == 0);
                false_block = basic_block(a, nodes(a, 0, NULL), unique_name(a, "if_false"));
                set_abstraction_body((Node*) false_block, join(a, (Join) { .join_point = jp, .args = nodes(a, 0, NULL), .mem = get_abstraction_mem(false_block) }));
            }

            Node* control_case = basic_block(a, singleton(jp), NULL);
            const Node* control_body = branch(a, (Branch) {
                .condition = rewrite_node(r, node->payload.if_instr.condition),
                .true_jump = jump_helper(a, true_block, empty(a), get_abstraction_mem(control_case)),
                .false_jump = jump_helper(a, false_block, empty(a), get_abstraction_mem(control_case)),
                .mem = get_abstraction_mem(control_case),
            });
            set_abstraction_body(control_case, control_body);

            BodyBuilder* bb = begin_body_with_mem(a, nmem);
            Nodes results = gen_control(bb, yield_types, control_case);
            return finish_body(bb, jump_helper(a, rewrite_node(r, payload.tail), results, bb_mem(bb)));
        }
        // TODO: match
        case Loop_TAG: {
            Loop payload = node->payload.loop_instr;
            const Node* old_loop_block = payload.body;

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, get_param_types(a, get_abstraction_params(old_loop_block)));
            param_types = strip_qualifiers(a, param_types);

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

            Nodes new_params = recreate_params(&ctx->rewriter, get_abstraction_params(old_loop_block));
            Node* loop_header_block = basic_block(a, new_params, unique_name(a, "loop_header"));

            BodyBuilder* inner_bb = begin_body_with_mem(a, get_abstraction_mem(loop_header_block));
            Node* inner_control_case = case_(a, singleton(continue_point));
            set_abstraction_body(inner_control_case, jump_helper(a, rewrite_node(r, old_loop_block), new_params, get_abstraction_mem(inner_control_case)));
            Nodes args = gen_control(inner_bb, param_types, inner_control_case);

            set_abstraction_body(loop_header_block, finish_body(inner_bb, jump(a, (Jump) { .target = loop_header_block, .args = args, .mem = bb_mem(inner_bb) })));

            Node* outer_control_case = case_(a, singleton(break_point));
            const Node* first_iteration_jump = jump(a, (Jump) {
                .target = loop_header_block,
                .args = rewrite_nodes(r, payload.initial_args),
                .mem = get_abstraction_mem(outer_control_case),
            });
            set_abstraction_body(outer_control_case, first_iteration_jump);

            BodyBuilder* bb = begin_body_with_mem(a, rewrite_node(r, payload.mem));
            Nodes results = gen_control(bb, yield_types, outer_control_case);
            return finish_body(bb, jump_helper(a, rewrite_node(r, payload.tail), results, bb_mem(bb)));
        }
        case MergeSelection_TAG: {
            MergeSelection payload = node->payload.merge_selection;
            const Node* root_mem = get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
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
            const Node* jp = first(*jps);
            assert(jp);
            const Node* nmem = rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem
            });
        }
        case MergeContinue_TAG: {
            MergeContinue payload = node->payload.merge_continue;
            const Node* root_mem = get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
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
            const Node* nmem = rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem,
            });
        }
        case MergeBreak_TAG: {
            MergeBreak payload = node->payload.merge_break;
            const Node* root_mem = get_original_mem(payload.mem);
            assert(root_mem->tag == AbsMem_TAG);
            CFNode* cfnode = cfg_lookup(ctx->cfg, root_mem->payload.abs_mem.abs);
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
            const Node* jp = first(*jps);
            assert(jp);
            const Node* nmem = rewrite_node(r, payload.mem);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, payload.args),
                .mem = nmem,
            });
        }
        default: break;
    }
    return recreate_node_identity(&ctx->rewriter, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

Module* lower_cf_instrs(SHADY_UNUSED const CompilerConfig* config, Module* src) {
    ArenaConfig aconfig = *get_arena_config(get_module_arena(src));
    IrArena* a = new_ir_arena(&aconfig);
    Module* dst = new_module(a, get_module_name(src));
    Context ctx = {
        .rewriter = create_node_rewriter(src, dst, (RewriteNodeFn) process_node),
        .structured_join_tokens = shd_new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    shd_destroy_dict(ctx.structured_join_tokens);
    return dst;
}

