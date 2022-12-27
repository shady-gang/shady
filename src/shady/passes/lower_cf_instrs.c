#include "passes.h"

#include "log.h"
#include "portability.h"
#include "../type.h"
#include "../rewrite.h"

#include <assert.h>

typedef struct {
    const Node* join_point_selection_merge;
    const Node* join_point_switch_merge;
    const Node* join_point_loop_break;
    const Node* join_point_loop_continue;
} JoinPoints;

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    JoinPoints join_points;
    Node* current_fn;
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* arena = ctx->rewriter.dst_arena;

    const Node* old_instruction = node->payload.let.instruction;
    const Node* new_instruction = NULL;
    const Node* old_tail = node->payload.let.tail;
    const Node* new_tail = NULL;

    switch (old_instruction->tag) {
        case If_TAG: {
            bool has_false_branch = old_instruction->payload.if_instr.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.if_instr.yield_types);

            const Type* jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Node* join_point = var(arena, jp_type, "if_join");
            Context join_context = *ctx;
            join_context.join_points.join_point_selection_merge = join_point;

            Node* true_block = basic_block(arena, ctx->current_fn, nodes(arena, 0, NULL), unique_name(arena, "if_true"));
            true_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, old_instruction->payload.if_instr.if_true->payload.anon_lam.body);

            Node* flse_block = basic_block(arena, ctx->current_fn, nodes(arena, 0, NULL), unique_name(arena, "if_false"));
            if (has_false_branch)
                flse_block->payload.basic_block.body = rewrite_node(&join_context.rewriter, old_instruction->payload.if_instr.if_false->payload.anon_lam.body);
            else {
                assert(yield_types.count == 0);
                flse_block->payload.basic_block.body = join(arena, (Join) { .join_point = join_point, .args = nodes(arena, 0, NULL) });
            }

            const Node* control_body = branch(arena, (Branch) {
                .branch_condition = rewrite_node(&ctx->rewriter, old_instruction->payload.if_instr.condition),
                .true_target = true_block,
                .false_target = flse_block,
                .args = nodes(arena, 0, NULL),
            });

            const Node* control_lam = lambda(ctx->rewriter.dst_module, nodes(arena, 1, (const Node*[]) {join_point }), control_body);
            new_instruction = control(arena, (Control) { .yield_types = yield_types, .inside = control_lam });
            break;
        }
        case Loop_TAG: {
            const Node* old_loop_body = old_instruction->payload.loop_instr.body;
            assert(is_anonymous_lambda(old_loop_body));

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, extract_variable_types(arena, old_loop_body->payload.anon_lam.params));

            const Type* break_jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = yield_types }),
                .is_uniform = true,
            });
            const Type* continue_jp_type = qualified_type(arena, (QualifiedType) {
                .type = join_point_type(arena, (JoinPointType) { .yield_types = param_types }),
                .is_uniform = true,
            });
            const Node* break_point = var(arena, break_jp_type, "loop_break_point");
            const Node* continue_point = var(arena, continue_jp_type, "loop_continue_point");
            Context join_context = *ctx;
            join_context.join_points.join_point_loop_break = break_point;
            join_context.join_points.join_point_loop_continue = continue_point;

            Nodes new_params = recreate_variables(&ctx->rewriter, old_loop_body->payload.anon_lam.params);
            Node* loop_body = basic_block(arena, ctx->current_fn, new_params, unique_name(arena, "loop_body"));
            register_processed_list(&join_context.rewriter, old_loop_body->payload.anon_lam.params, loop_body->payload.basic_block.params);

            const Node* inner_control_body = rewrite_node(&join_context.rewriter, old_loop_body->payload.anon_lam.body);
            const Node* inner_control_lam = lambda(ctx->rewriter.dst_module, nodes(arena, 1, (const Node*[]) {continue_point }), inner_control_body);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            const Node* inner_control = control(arena, (Control) {
                .yield_types = param_types,
                .inside = inner_control_lam,
            });
            Nodes args = bind_instruction(bb, inner_control);

            // TODO let_in_block or use a Jump !
            loop_body->payload.basic_block.body = finish_body(bb, jump(arena, (Jump) { .target = loop_body, .args = args }));

            const Node* initial_jump = jump(arena, (Jump) {
                .target = loop_body,
                .args = rewrite_nodes(&ctx->rewriter, old_instruction->payload.loop_instr.initial_args),
            });
            const Node* outer_body = lambda(ctx->rewriter.dst_module, nodes(arena, 1, (const Node*[]) { break_point }), initial_jump);
            new_instruction = control(arena, (Control) { .yield_types = yield_types, .inside = outer_body });
            break;
        }
        default:
            new_instruction = rewrite_node(&ctx->rewriter, old_instruction);
            break;
    }

    if (!new_tail)
        new_tail = rewrite_node(&ctx->rewriter, old_tail);

    assert(new_instruction && new_tail);
    return let(arena, new_instruction, new_tail);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    if (node->tag == Function_TAG) {
        Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
        Context sub_ctx = *ctx;
        sub_ctx.disable_lowering = lookup_annotation_with_string_payload(fun, "DisablePass", "lower_cf_instrs");
        sub_ctx.current_fn = fun;
        sub_ctx.join_points = (JoinPoints) {
            .join_point_selection_merge = NULL,
            .join_point_switch_merge = NULL,
            .join_point_loop_break = NULL,
            .join_point_loop_continue = NULL,
        };
        fun->payload.fun.body = rewrite_node(&sub_ctx.rewriter, node->payload.fun.body);
        return fun;
    }

    if (ctx->disable_lowering)
        return recreate_node_identity(&ctx->rewriter, node);

    switch (node->tag) {
        case Let_TAG: return process_let(ctx, node);
        case MergeSelection_TAG: {
            const Node* jp = ctx->join_points.join_point_selection_merge;
            assert(jp);
            return join(dst_arena, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_selection.args),
            });
        }
        case MergeContinue_TAG: {
            const Node* jp = ctx->join_points.join_point_loop_continue;
            assert(jp);
            return join(dst_arena, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_continue.args),
            });
        }
        case MergeBreak_TAG: {
            const Node* jp = ctx->join_points.join_point_loop_break;
            assert(jp);
            return join(dst_arena, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_break.args),
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

void lower_cf_instrs(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process_node),
    };
    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}

