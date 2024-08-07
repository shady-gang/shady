#include "pass.h"

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
    const Node* abs;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* process_instruction(Context* ctx, const Node* old_instruction) {
    IrArena* a = ctx->rewriter.dst_arena;

    switch (old_instruction->tag) {
        default:
            break;
    }

    return recreate_node_identity(&ctx->rewriter, old_instruction);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    Rewriter* r = &ctx->rewriter;
    IrArena* a = r->dst_arena;

    Context sub_ctx = *ctx;
    if (node->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        sub_ctx.disable_lowering = lookup_annotation(fun, "Structured");
        sub_ctx.current_fn = fun;
        sub_ctx.cfg = build_fn_cfg(node);
        sub_ctx.abs = node;
        fun->payload.fun.body = rewrite_node(&sub_ctx.rewriter, node->payload.fun.body);
        destroy_cfg(sub_ctx.cfg);
        return fun;
    } else if (node->tag == Constant_TAG) {
        sub_ctx.cfg = NULL;
        sub_ctx.abs = NULL;
        sub_ctx.current_fn = NULL;
        ctx = &sub_ctx;
    }

    if (is_abstraction(node)) {
        sub_ctx.abs = node;
        ctx = &sub_ctx;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    CFNode* cfnode = ctx->cfg ? cfg_lookup(ctx->cfg, ctx->abs) : NULL;
    if (is_instruction(node))
        return process_instruction(ctx, node);
    switch (node->tag) {
        case If_TAG: {
            bool has_false_branch = node->payload.if_instr.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.if_instr.yield_types);
            const Node* nmem = rewrite_node(r, node->payload.if_instr.mem);

            const Type* jp_type = qualified_type(a, (QualifiedType) {
                .type = join_point_type(a, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = false,
            });
            const Node* jp = param(a, jp_type, "if_join");
            Context join_context = *ctx;
            Nodes jps = singleton(jp);
            insert_dict(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            Node* true_block = basic_block(a, nodes(a, 0, NULL), unique_name(a, "if_true"));
            join_context.abs = node->payload.if_instr.if_true;
            true_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, get_abstraction_body(node->payload.if_instr.if_true));

            Node* flse_block = basic_block(a, nodes(a, 0, NULL), unique_name(a, "if_false"));
            if (has_false_branch) {
                join_context.abs = node->payload.if_instr.if_false;
                flse_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, get_abstraction_body(node->payload.if_instr.if_false));
            } else {
                assert(yield_types.count == 0);
                flse_block->payload.basic_block.body = join(a, (Join) { .join_point = jp, .args = nodes(a, 0, NULL) });
            }

            BodyBuilder* bb = begin_body_with_mem(a, nmem);
            Node* control_case = basic_block(a, singleton(jp), NULL);
            const Node* control_body = branch(a, (Branch) {
                .condition = rewrite_node(r, node->payload.if_instr.condition),
                .true_jump = jump_helper(a, true_block, empty(a), get_abstraction_mem(control_case)),
                .false_jump = jump_helper(a, flse_block, empty(a), get_abstraction_mem(control_case)),
                .mem = get_abstraction_mem(control_case),
            });
            set_abstraction_body(control_case, control_body);
            Nodes results = gen_control(bb, yield_types, control_case);

            const Node* otail = node->payload.if_instr.tail;
            Node* join = basic_block(a, recreate_params(r, get_abstraction_params(otail)), NULL);
            register_processed_list(r, get_abstraction_params(otail), get_abstraction_params(join));
            set_abstraction_body(join, rewrite_node(r, get_abstraction_body(otail)));
            return finish_body(bb, jump_helper(a, join, results, bb_mem(bb)));
            // return control(a)
            //return yield_values_and_wrap_in_block(bb, );
        }
        // TODO: match
        case Loop_TAG: {
            const Node* old_loop_body = node->payload.loop_instr.body;

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, node->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, get_param_types(a, get_abstraction_params(old_loop_body)));
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
            Context join_context = *ctx;
            Nodes jps = mk_nodes(a, break_point, continue_point);
            insert_dict(const Node*, Nodes, ctx->structured_join_tokens, node, jps);

            Nodes new_params = recreate_params(&ctx->rewriter, get_abstraction_params(old_loop_body));
            Node* loop_body = basic_block(a, new_params, unique_name(a, "loop_body"));
            register_processed_list(&join_context.rewriter, get_abstraction_params(old_loop_body), loop_body->payload.basic_block.params);

            join_context.abs = old_loop_body;
            const Node* inner_control_body = rewrite_node(&join_context.rewriter, get_abstraction_body(old_loop_body));
            Node* inner_control_case = case_(a, singleton(continue_point));
            set_abstraction_body(inner_control_case, inner_control_body);

            BodyBuilder* inner_bb = begin_body_with_mem(a, get_abstraction_mem(inner_control_case));
            Nodes args = gen_control(inner_bb, param_types, inner_control_case);

            // TODO let_in_block or use a Jump !
            loop_body->payload.basic_block.body = finish_body(inner_bb, jump(a, (Jump) { .target = loop_body, .args = args }));

            const Node* initial_jump = jump(a, (Jump) {
                .target = loop_body,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.loop_instr.initial_args),
            });

            Node* outer_control_case = case_(a, singleton(break_point));
            BodyBuilder* outer_bb = begin_body_with_mem(a, get_abstraction_mem(outer_control_case));
            Nodes results = gen_control(outer_bb, yield_types, outer_control_case);
            set_abstraction_body(outer_control_case, initial_jump);

            const Node* otail = get_structured_construct_tail(node);
            Node* join = basic_block(a, recreate_params(r, get_abstraction_params(otail)), NULL);
            register_processed_list(r, get_abstraction_params(otail), get_abstraction_params(join));
            set_abstraction_body(join, rewrite_node(r, get_abstraction_body(otail)));
            return finish_body(outer_bb, jump_helper(a, join, results, bb_mem(outer_bb)));
        }
        case MergeSelection_TAG: {
            if (!cfnode)
                break;
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
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 1);
            const Node* jp = first(*jps);
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_selection.args),
                .mem = rewrite_node(r, node->payload.merge_selection.mem)
            });
        }
        case MergeContinue_TAG: {
            assert(cfnode);
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                if (body->tag == Loop_TAG) {
                    selection_instr = body;
                    break;
                }
                dom = dom->idom;
            }

            if (!selection_instr) {
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 2);
            const Node* jp = jps->nodes[1];
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_continue.args),
                .mem = rewrite_node(r, node->payload.merge_continue.mem)
            });
        }
        case MergeBreak_TAG: {
            assert(cfnode);
            CFNode* dom = cfnode->idom;
            const Node* selection_instr = NULL;
            while (dom) {
                const Node* body = get_abstraction_body(dom->node);
                if (body->tag == Loop_TAG) {
                    selection_instr = body;
                    break;
                }
                dom = dom->idom;
            }

            if (!selection_instr) {
                error_print("Scoping error: Failed to find a dominating selection construct for ");
                log_node(ERROR, node);
                error_print(".\n");
                error_die();
            }

            Nodes* jps = find_value_dict(const Node*, Nodes, ctx->structured_join_tokens, selection_instr);
            assert(jps && jps->count == 2);
            const Node* jp = first(*jps);
            assert(jp);
            return join(a, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_break.args),
                .mem = rewrite_node(r, node->payload.merge_break.mem)
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
        .structured_join_tokens = new_dict(const Node*, Nodes, (HashFn) hash_node, (CmpFn) compare_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
    destroy_dict(ctx.structured_join_tokens);
    return dst;
}

