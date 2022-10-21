#include "shady/ir.h"

#include "log.h"
#include "portability.h"
#include "../type.h"
#include "../rewrite.h"

#include "list.h"

#include "dict.h"

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
} Context;

static const Node* process_node(Context* ctx, const Node* node);

static const Node* process_let(Context* ctx, const Node* node) {
    assert(node->tag == Let_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    const Node* instr = node->payload.let.instruction;
    const Node* ninstr = NULL;
    const Node* old_tail = node->payload.let.tail;
    const Node* new_tail = NULL;

    switch (instr->tag) {
        case If_TAG: {
            bool has_false_branch = instr->payload.if_instr.if_false;
            Nodes yield_types = rewrite_nodes(&ctx->rewriter, instr->payload.if_instr.yield_types);

            const Node* join_point = var(dst_arena, join_point_type(dst_arena, (JoinPointType) { .yield_types = yield_types }), "if_join");
            Context join_context = *ctx;
            join_context.join_points.join_point_selection_merge = join_point;

            Node* true_block = basic_block(dst_arena, nodes(dst_arena, 0, NULL), unique_name(dst_arena, "if_true"));
            true_block->payload.lam.body = rewrite_node(&join_context.rewriter,  instr->payload.if_instr.if_true);

            Node* flse_block = basic_block(dst_arena, nodes(dst_arena, 0, NULL), unique_name(dst_arena, "if_false"));
            if (has_false_branch)
                flse_block->payload.lam.body = rewrite_node(&join_context.rewriter,  instr->payload.if_instr.if_false);
            else {
                assert(yield_types.count == 0);
                flse_block->payload.lam.body = join(dst_arena, (Join) { .join_point = join_point, .args = nodes(dst_arena, 0, NULL) });
            }

            const Node* branch_t = branch(dst_arena, (Branch) {
                .branch_mode = BrIfElse,
                .branch_condition = rewrite_node(&ctx->rewriter, instr->payload.if_instr.condition),
                .true_target = true_block,
                .false_target = flse_block,
                .args = nodes(dst_arena, 0, NULL),
            });

            Node* if_body = lambda(dst_arena, nodes(dst_arena, 1, (const Node*[]) { join_point }));
            if_body->payload.lam.body = branch_t;
            ninstr = control(dst_arena, (Control) { .yield_types = yield_types, .inside = if_body });
            break;
        }
        case Loop_TAG: {
            const Node* old_loop_body = instr->payload.loop_instr.body;

            Nodes yield_types = rewrite_nodes(&ctx->rewriter, instr->payload.loop_instr.yield_types);
            Nodes param_types = rewrite_nodes(&ctx->rewriter, extract_variable_types(dst_arena, &instr->payload.lam.params));

            const Node* break_point = var(dst_arena, join_point_type(dst_arena, (JoinPointType) { .yield_types = yield_types }), "loop_break_point");
            const Node* continue_point = var(dst_arena, join_point_type(dst_arena, (JoinPointType) { .yield_types = param_types }), "loop_continue_point");
            Context join_context = *ctx;
            join_context.join_points.join_point_loop_break = break_point;
            join_context.join_points.join_point_loop_continue = continue_point;

            Nodes new_params = recreate_variables(&ctx->rewriter, old_loop_body->payload.lam.params);
            Node* loop_body = basic_block(dst_arena, new_params, unique_name(dst_arena, "loop_body"));
            register_processed_list(&join_context.rewriter, old_loop_body->payload.lam.params, loop_body->payload.lam.params);

            Node* actual_body = lambda(dst_arena, nodes(dst_arena, 1, (const Node*[]) { continue_point }));
            actual_body->payload.lam.body = rewrite_node(&join_context.rewriter, old_loop_body->payload.lam.body);

            const Node* inner_control = control(dst_arena, (Control) {
                .yield_types = param_types,
                .inside = actual_body,
            });

            loop_body->payload.lam.body = let(dst_arena, false, inner_control, loop_body);

            Node* outer_body = lambda(dst_arena, nodes(dst_arena, 1, (const Node*[]) { break_point }));
            const Node* initial_jump = branch(dst_arena, (Branch) {
                .branch_mode = BrJump,
                .target = loop_body,
                .args = rewrite_nodes(&ctx->rewriter, instr->payload.loop_instr.initial_args),
            });
            outer_body->payload.lam.body = initial_jump;
            ninstr = control(dst_arena, (Control) { .yield_types = yield_types, .inside = outer_body });
        }
        default: break;
    }

    if (!new_tail)
        new_tail = rewrite_node(&ctx->rewriter, old_tail);

    assert(ninstr && new_tail);
    return let(dst_arena, false, ninstr, new_tail);
}

static const Node* process_node(Context* ctx, const Node* node) {
    const Node* already_done = search_processed(&ctx->rewriter, node);
    if (already_done)
        return already_done;

    IrArena* dst_arena = ctx->rewriter.dst_arena;

    switch (node->tag) {
        case Lambda_TAG: {
            Node* fun = recreate_decl_header_identity(&ctx->rewriter, node);
            Context sub_ctx = *ctx;
            if (node->payload.lam.tier == FnTier_Function) {
                sub_ctx.disable_lowering = lookup_annotation_with_string_payload(fun, "DisablePass", "lower_cf_instrs");
                sub_ctx.join_points = (JoinPoints) {
                    .join_point_selection_merge = NULL,
                    .join_point_switch_merge = NULL,
                    .join_point_loop_break = NULL,
                    .join_point_loop_continue = NULL,
                };
            }
            fun->payload.lam.body = rewrite_node(&sub_ctx.rewriter, node->payload.lam.body);
            return fun;
        }
        case Let_TAG: return process_let(ctx, node);
        case MergeConstruct_TAG: {
            const Node* jp = NULL;
            switch (node->payload.merge_construct.construct) {
                case Selection: jp = ctx->join_points.join_point_selection_merge; break;
                case Continue:  jp = ctx->join_points.join_point_loop_continue; break;
                case Break:     jp = ctx->join_points.join_point_loop_break; break;
            }
            assert(jp);
            return join(dst_arena, (Join) {
                .join_point = jp,
                .args = rewrite_nodes(&ctx->rewriter, node->payload.merge_construct.args),
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, node);
    }
}

const Node* lower_cf_instrs(SHADY_UNUSED CompilerConfig* config, IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    Context ctx = {
        .rewriter = create_rewriter(src_arena, dst_arena, (RewriteFn) process_node),
        .disable_lowering = false,
    };

    assert(src_program->tag == Root_TAG);

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);

    destroy_rewriter(&ctx.rewriter);
    return rewritten;
}
